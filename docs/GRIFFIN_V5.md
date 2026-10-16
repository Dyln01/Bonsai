# GRIFFIN v5 — sequence-level neuron selection (training-free, calibration-free)

**Date:** 2026-10-15 · engine `qwen36` @ [all prior] + **GRIFFIN v5** · no downloads.

## What was implemented (RESEARCH_MAP item #1)

From Dong et al. 2024 ("flocking"): tokens within a sequence activate similar FF neurons, so
**the prompt itself names the neuron set** — no predictor training, no corpus calibration.

Mechanics, wired into the paged MLP path:
1. **Observe (prefill):** the stock batched MLP path accumulates per-neuron Σ|gate| over all
   prompt positions (`q36_griff_observe`, per-layer double accumulators).
2. **Finalize (once):** top-K per layer by observed importance, K = HOT_FRAC·I — the SAME quality
   dial; indices kept ascending for contiguity (`q36_griff_finalize_all`).
3. **Decode:** when a set exists and K < I, the gather path computes exactly those neurons:
   gate/up as per-row mapped reads (row-view QWs over the mmap), down as a column gather per
   output row — original index order, so any container works, sorted or not.

Opt-in: `GRIFFIN=1`; without it the prefix path is untouched. `HOT_FRAC` still controls K.

## Measured (calibrated int4 student, 4 threads, mapped trunk)

| arm | rc | TTFT (2.4k-tok prompt) | logits vs full ref |
|---|---|---|---|
| frac 1.0 (reference) | 0 | 26.5 s | — |
| prefix cut 0.25 | 0 | 11.6 s | rel-err 0.0000 |
| **GRIFFIN 0.25** | 0 | **12.1 s** | **rel-err 0.0000** |

- The gather path is exercised end-to-end (prefill observation + finalize + selected-neuron decode).
- ~4% prefill overhead vs the prefix cut (observation pass + row-view construction) — decode-side
  both compute K neurons.
- On this synthetic, BOTH cuts are print-precision-identical to the full model: the measured
  energy concentration (99.6% in top-25%) puts the dropped 75% below f32 visibility at the logits
  (the synthetic's logits sit at ~1e-36 scale; per-element absmax diff 1.07e-35 with vector
  rel-err < 5e-5). **The discriminating experiment — griffin beats a stale prefix when the
  workload drifts from calibration — needs logits at sane scale, i.e. real weights.** The
  machinery is ready for that run unchanged.

## Why it matters despite the synthetic tie

The prefix cut's one weakness: if the serving workload drifts from the calibration corpus, the
prefix is stale. GRIFFIN re-derives the set from *every prompt* — self-adapting, zero maintenance.
Cost: ~4% prefill. On the 27B projection: same ~12-14 tok/s warm decode as prefix paging, but
robust to workload drift; and it works on UNMODIFIED containers (no hot-sort step needed at all).

## How to use

```bash
# any int4/f16 container (sorted or not), full-size dense model, ~K% of MLP computed:
TRUNK_MAP=1 GRIFFIN=1 HOT_FRAC=0.25 TRUNK_WARM=1 KV8=1 SNAP=<container> ./qwen36 32 8 prompt.txt
# HOT_FRAC=1.0 disables any cutting (griffin set k=I -> stock path).
```

## Files

- Engine: `~/colibri/c/qwen36.c` (builds clean; backups `.premap`/`.prehot` still valid)
- Patcher: `colibri-harness/patch_griffin.py` (anchors: HotAcc block, stock batched silu,
  paging decision, profile-save seam)
- Gates: `col_griffgates.sh` / `col_drift.sh` patterns (calibrated + unsorted containers)
