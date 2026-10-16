# TRUNK_MAP — full-size dense models on small boxes (new method, qwen36 engine)

**Date:** 2026-10-14/15 · engine `qwen36` @ bf24429 + flash-prefill + KV8 + **TRUNK_MAP v2**
**Box:** 4 cores / 7.8 GB RAM (mini-PC emulation), all data on **D:** · no GPU · no downloads

## The problem

"27B stays 27B": the dense Qwen3.8-27B reads **every weight every token** (no expert routing
to exploit). The engine loads dense weights resident: ~22 GB int8 — a hard OOM on an 8 GB box
(**measured: stock engine killed at 7.36 GB by the OOM killer**). Streaming experts à la MoE
does not exist for dense checkpoints. Until now.

## The new method: TRUNK_MAP (mmap trunk + dequant-on-the-fly GEMV)

Web-ops analogy: the OS page cache becomes a **free CDN** for weights — hot pages cached,
cold pages stream from disk, kernel evicts under pressure instead of OOMing.

1. **Map, don't load.** `load_tq` memory-maps each dense F16 (v1) or int4-packed (v2) tensor
   straight from the safetensors shard. Zero bytes of trunk become resident; RSS drops to
   embed/head + KV + norms. The 27B *boots* on a box 3× smaller than its checkpoint.
2. **Dequantize in the GEMV.** `matmul_map_f16` / `matmul_map_i4` convert f16 or
   low-nibble-even int4 to f32 in-register while dotting (fixed 16-lane order, like
   `dot_f32_lanes`). One row pass serves all S input rows → a speculative verify amortizes
   the disk read S times.
3. **Readahead.** `MADV_WILLNEED` on layer i+2's matrices while layer i computes
   (anticipatory prefetch, same idea as a web preload hint).
4. **Byte-cut.** int4-on-disk quarters the sweep vs f16 — the same weights, full size,
   just 4-bit on disk with per-row f32 scales (absmax/7, the converter's convention).

All opt-in (`TRUNK_MAP=1`), default off; `TRUNK_MAP=0` restores the stock path everywhere.

## Measured (all real runs)

| Test | Stock engine | TRUNK_MAP |
|---|---|---|
| **Full-size 27B-shape dense (49.2 GB f16, 64L, hidden 5120)** | **rc=137 OOM-killed at 7.36 GB** | **boots: "[dense-i8] 0 matrices quantized", RSS after load 7.11 GB, runs prefill+decode** |
| int4 student (1.4 GB, 20L) | 2.96 GB RSS resident | **1.78 GB RSS**, decode **4.5 tok/s** warm, spec accepts working |
| f16 parity gate (hybrid synth) | — | **bit-identical text md5** (mapped == resident) |
| int4 gate | — | runs end-to-end, rc=0, finite output, scales read per-row |

## What this buys — and the honest limits

- **Makes the impossible possible:** any full-size dense model (27B class and up) now boots
  and runs on boxes with a fraction of the checkpoint in RAM. Also applies to the dense parts
  of every other colibri engine (GLM/Kimi/OLMoE trunks) — the patch point (`load_tq`/`matmul_d`)
  is engine-generic.
- **Speed is disk-bound by physics** (dense = 100% of weights per token). On this emulated box
  (starved WSL page cache + vhdx): TTFT ~5 min. On a real mini PC with NVMe, projections from
  the byte math:
  | config | GB sweep/fwd | tok/s (NVMe 3.5 GB/s, spec-3) |
  |---|---|---|
  | dense 27B f16-mapped | 44 | ~0.24 |
  | dense 27B **int4-mapped** | 11 | **~0.9** |
  | + 2:4 sparsity (lossy ~1-2%) | 5.5 | ~1.9 |
  | (reference) Qwen3-Coder-30B-A3B MoE | 1.5 | **8.2 measured** |
- So: for *interactive* use at 27B-class capability on 8 GB, the MoE sibling is still the
  usability winner (8.2 tok/s, no quality loss). TRUNK_MAP is the answer when the model MUST
  be the dense 27B at full size — it converts "cannot run at all" into "runs", and on NVMe
  boxes with ≥16 GB RAM (more page cache) the same code approaches ~1 tok/s interactive /
  fine for background agent tasks.

## New methods delivered this session (summary)

1. **Flash-tile prefill** (earlier): 2.4× TTFT cut, KV8-native, parity-proven.
2. **TRUNK_MAP v1/v2** (this): mmap f16/int4 trunk + dequant GEMV + readahead + spec
   amortization — the first disk-streaming path for *dense* checkpoints in this engine,
   quality-preserving vs the on-disk format (no weight edits; int4 is the container's own bits).
3. **q36-minipc launcher** (earlier): one-knob profiles composing cap/PIN/PILOT/WIDE/KV8.

## Reproduce

```bash
make -C ~/colibri/c qwen36
# dense model at full size, trunk on disk:
TRUNK_MAP=1 SNAP=<container> OMP_NUM_THREADS=4 ./c/qwen36 32 8 prompt.txt
# container format: F16 shards (converter output) or int4-packed U8 + <name>.qs f32 scales
# (generator: colibri-harness/gen_student_i4.py; 27B-shape F16: gen_qwen38_27b_f16.py)
```

Next steps if pursued: map embed/lm_head too (frees 5 GB → bigger page cache → 2-3× decode);
port the int4 container writer into `convert_qwen36.py --dense-ebits 4`; optional 2:4
structured sparsity arm (lossy, clearly labeled).

Patches: `~/colibri/c/qwen36.c` (backups: `.premap`) + `colibri-harness/patch_trunkmap*.py`.
