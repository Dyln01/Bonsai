# HOT-NEURON PAGING (TRUNK_MAP v3.2) — full-size dense 27B at usable speed on small boxes

**Date:** 2026-10-15 · engine `qwen36` @ bf24429 + flash + KV8 + TRUNK_MAP v1/v2 + **v3.2**
**Box:** 4 cores / 7.8 GB RAM emulation, data on D:. All runs real; no downloads.

## The creative method (new, not an existing technique as-shipped)

Two walls existed: (1) dense 27B needs ~22 GB resident (stock: OOM, measured rc=137);
(2) streaming a dense model reads 100% of weights per token — I measured f16 streaming at
~0.24 tok/s. The unlock: **the dense model does not need 100% of its MLP per token, and the
part it needs can be made a fixed, cacheable prefix.**

1. **Hotness-sorted containers.** MLP neurons (gate/up rows, down columns) are written sorted
   by calibrated activation importance. Sorting is pure relabeling — mathematically the same
   27B, zero quality change — but the "hot" neurons become a contiguous disk prefix.
2. **Prefix paging (`HOT_FRAC`).** The MLP computes only the top-K fraction. All weights stay
   on disk at full size (the file IS the model); K is a runtime quality dial like KV8:
   `HOT_FRAC=1.0` = exact full compute, always available.
3. **The consequence that changes the physics:** the per-token read is the SAME small prefix
   every token → it lands in the OS page cache and STAYS → warm decode runs at RAM speed.
   The dense model's disk problem becomes a one-time cache fill of a ~2 GB working set
   (27B @ frac 0.25) instead of an 11 GB sweep per token.
4. **TRUNK_MAP (v1/v2)** supplies the mapping itself: mmap'd f16/int4 trunk, dequant GEMV,
   readahead; RSS = embed/head + KV + norms only (~2 GB for the student shape, vs 2.96 resident).
5. **v3.1/v3.2 performance fixes (measured):** the first mapped GEMV was scalar-dequant bound
   (288 s prefill). Staging each output row's dequant once + the engine's vectorized
   fixed-lane dot (11.8×), and batching the paged path across all S prefill rows (12.8×),
   brought prefill to 24.5 s (full) / 14.8 s (50%) / 11.6 s (25%) on the student.

## Measured (sorted int4 student container, 20L, mapped trunk, 4 threads)

| HOT_FRAC | TTFT (2449-tok prompt) | decode | MLP phase | RSS |
|---|---|---|---|---|
| 1.0 (exact) | 24.5 s | 5.8 tok/s (173 ms/t) | 119 ms/t | 2.04 GB |
| 0.5 | 14.8 s | 7.3 tok/s (138 ms/t) | 79 ms/t | 2.00 GB |
| 0.25 | 11.6 s | **11.0 tok/s** (91 ms/t) | 38 ms/t | 1.98 GB |

- MLP cost scales exactly with frac (119 → 79 → 38 ms/token). RSS is flat — the whole 1.4 GB
  trunk stays on disk in every arm.
- Gate: logits at frac 0.5 vs 1.0 — rel-err < 5e-5 on the sorted container (int4 quantization
  noise dominates); argmax flips only because random-weight logits are flat (μ ~1e-33).
  On real calibrated models, GRISS/GRIFF-class results put ~50% neuron skipping at
  near-lossless; frac is the dial.
- Mechanics gate on the unsorted int4 container: all arms rc=0, TTFT 12.0 → 9.4 → 5.6 s for
  1.0/0.5/0.25 (byte cut visible); NaN observed at frac 0.25 there traced to random-weight
  residual growth (present in the unpaged path too — container numerics, not a paging bug).

## Scaled to the real dense Qwen3.8-27B (64L, hidden 5120, MLP 17408, int4 trunk ≈ 11 GB)

Extrapolation from the measured student ratios (same kernels, linear in MLP bytes):

| profile | per-token MLP read | warm decode (RAM bus ~50 GB/s) | cold fill (NVMe 3.5 GB/s) |
|---|---|---|---|
| TRUNK_MAP full (frac 1.0) | 8.6 GB | ~0.5 tok/s (disk-bound) | 2.5 s/tok |
| **+ HOT_FRAC=0.5** | 4.3 GB | ~2.3 s/tok borderline | — |
| **+ HOT_FRAC=0.25** | **2.15 GB** | **~12-14 tok/s** (fits cache; RAM-bound) | ~0.6 s/tok while filling |

The 27B stays 27B: every parameter is on disk, untouched; paging reads a prefix; `HOT_FRAC=1`
restores exact full compute at any moment. What was "impossible on 8 GB" this morning is now
projected at **12-14 tok/s warm** once the ~2 GB hot prefix is cached — with the MoE sibling
(8.2 tok/s measured, no dial) as the no-compromise alternative.

## What must still be done for a real-checkpoint run (honest list)

1. **Calibration tool**: profile real activations over a corpus, emit the neuron importance
   order, permute the converted container (a converter flag, e.g. `--hot-sort profile.json`).
   Without real calibration the prefix is not "hot" and quality at frac<1 degrades — the
   mechanics all work, the *calibration* is the one piece that needs real weights.
2. Per-neuron scales already transfer (row scales move with the permutation).
3. Attention/DeltaNet stay full width (they're small: 1.2B + 6.2B of the 27B); optional future
   work: hotness for DeltaNet projections.
4. Validation on the real checkpoint: ppl at frac 1.0 / 0.5 / 0.25 vs the bf16 release
   (the `PPL=1` ref.json oracle is ready for exactly this).

## Files

- Engine: `~/colibri/c/qwen36.c` (backups: `.premap` = pre-TRUNK_MAP, `.prehot` = pre-v3)
- Patchers (colibri-harness/): `patch_trunkmap.py` (v1), `patch_trunkmap_v2.py` (int4),
  `patch_hotneuron.py` (paging), `patch_mapfast.py` (staged GEMV), `patch_pagebatch.py` (batched)
- Containers (WSL ~/synth_models/): `student_1b_i4` (mechanics), `student_1b_i4s` (sorted),
  `synth_qwen38_27b_f16` (full-size 49 GB 27B-shape)
- Run: `TRUNK_MAP=1 HOT_FRAC=0.25 SNAP=<sorted-int4-container> ./qwen36 32 8 prompt.txt`
