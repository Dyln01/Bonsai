# qwen36 flash-prefill (Q36_FLASH) — measured results

**Date:** 2026-10-13 · engine `qwen36` @ bf24429 + KV8 patch v3 + **flash-prefill patch v2** (`qwen36.c`, backup `qwen36.c.preflash`)
**Model:** synthetic Qwen3.8-27B-shaped container (24L, 6 GQA + 18 DeltaNet, hidden 1024) — no download.
**Host:** WSL2, Ryzen 7 7735HS (8C/16T), 16 GB RAM. GPU: zero (CPU-only paths throughout).

## What was found (the real bottleneck)

The engine's `Speed:` line divides generated tokens by **total** wall time — prefill included. Splitting the phases with `COLI_TIMERS=1` showed decode was never the problem; **prefill attention was**:

| ctx (real tokens) | TTFT (prefill) | decode phase only |
|---|---|---|
| 2,449 | 4.3 s | 7.3 ms/tok → **137 tok/s** |
| 11,177 | 35.9 s | 10.3 ms/tok → **97 tok/s** |
| 23,465 | 151.4 s | 16.4 ms/tok → **61 tok/s** |

Prefill attention ran at **16.5 GFLOP/s** on a host whose raw dot floor is 86 GFLOP/s: the per-(head,row) loop re-read every K row once per query row and paid a 32-lane reduction tree per 64-element dot (`attnbench.c`, arms A vs E).

## What was built: `Q36_FLASH` — GQA head-grouped flash tiles for prefill attention (v2)

- Query blocks of QB=64 rows (retunable `Q36_FLASH_QB`), K/V tiles of TB=128 rows (`Q36_FLASH_TB`); online softmax, fused V accumulation; **each K/V tile is loaded once and serves every head of its kv-head and every row of the block** (`attnbench` arm C: 40 GFLOP/s, 2.4× the old layout).
- **KV8-native**: tiles dequantize once into per-thread staging; per-row Ksc/Vsc scales applied exactly as the original KV8 path (parity-proven). The low-RAM KV8 profile now gets flash too.
- Engages only for **S>8 prefill rows**; decode (S=1) and speculative verifies (S≤8) keep the original per-row path bit-for-bit; ref.json oracle mode never takes it (`g_q36_ref_mode`). `Q36_FLASH=0` restores the old path everywhere.
- Bugs caught by the gates: (1) KV8 parity md5 mismatch → missing per-row scales; (2) segfault at 23.5k ctx → `m->Kscale[layer]` loaded through NULL in f32 mode (fault addr 0x18 = layer 3×8). Both fixed; all parity gates pass.

## Measured matrix (all real engine runs, greedy, 8 threads)

**Battery point** (the original `exp_qwen27_final.py` workload — prompt_8k.txt = 15,960 byte-tokens, 64 new tokens):

| Arm | Peak RSS | TTFT before | TTFT now | decode phase |
|---|---|---|---|---|
| A. defaults (f32 KV, int8 dense) | 825 MB | 74.2 s | **56.9 s** | 12.0 ms/tok → 83 tok/s |
| B. KV8 | 690 MB | 112.2 s | **51.0 s** | 18.0 ms/tok → 56 tok/s |
| C. int4 dense | 747 MB | 76.6 s | **59.0 s** | 11.0 ms/tok → 91 tok/s |
| D. KV8 + int4 dense | **611 MB** | 112.3 s | **58.7 s** | 15.9 ms/tok → 63 tok/s |

**Long decode (N_NEW=256) at 15,960-token ctx:** f32 **85 tok/s** wall (78 tok/s over plain steps), KV8 identical.

**23,465-token ctx (prompt_distinct_4096):** TTFT 151→110 s (f32) and **220→91.5 s (KV8, 2.4×)**; decode 43.5 tok/s f32 (attention-read bound at S=1 — flash cannot help a single query row; that residual is the O(T) KV scan).

**Scaled to real Qwen3.8-27B** (16 attn layers, kv_heads 8, head_dim 128, hidden 5120): prefill attention is the same O(S²·H·hd) shape with 4× the per-head work; the tile reuse win is geometry-driven (K row serves 4 query heads instead of 2 here), so the same 1.6–2.4× TTFT cut and the KV8 flip (low-RAM arm becomes *fastest*) are the expected transfers. **Extrapolation from harness ratios; needs real-checkpoint confirmation.**

## Correctness gates (all PASS)

- f32 parity: generated text byte-identical flash ON vs OFF at 2.4k, 11k, and 23.5k ctx (md5).
- KV8 parity: generated text identical flash ON vs OFF at 23.5k (KV8-vs-f32 difference is the knob's own documented lossiness, unchanged).
- Original decode/spec-verify path untouched (S≤8 never enters flash); `Q36_FLASH=0` A/B available.
- rc=0 all arms; no RSS regression (flash staging is KBs per thread).

## How to run

```bash
# engine default: flash ON for prefill
env SNAP=<dir> N_NEW=64 ./qwen36 32 8 prompt.txt
# A/B: Q36_FLASH=0 ; tiles: Q36_FLASH_QB=64 Q36_FLASH_TB=128 ; debug: Q36_FLASH_DEBUG=1
# phase split: COLI_TIMERS=1  (attn split: proj | core | tail)
```

## Remaining known gaps

1. Decode at 23k+ ctx is O(T) per token (43.5 tok/s) — the flash trick does not apply at S=1; an MLA-style absorbed score or paged/quantized KV read is the next lever (KV8 already halves the bytes: decode 37.6 tok/s there).
2. Engine `Speed:` line still conflates prefill+decode — read `COLI_TIMERS=1` decode-phase ms/token for the honest rate (a reporting patch is a 5-line follow-up).
3. `exp_qwen27_final.py` used `NGEN` which the engine ignores (it reads `N_NEW`); every historical "48-token" battery actually ran 64.
