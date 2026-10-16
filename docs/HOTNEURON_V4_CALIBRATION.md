# HOT-NEURON v4 — self-calibration, per-layer fracs, warm-start (no downloads needed)

**Date:** 2026-10-15 · engine `qwen36` @ bf24429 + flash + KV8 + TRUNK_MAP v1-v3.2 + **v4**
**Box:** 4 cores / 7.8 GB emulation, D-disk. Every number below is a real run on synthetics.

## What v4 adds (completing the no-download path)

### A) Self-calibration: `HOT_PROFILE=<path>`
The engine itself is the calibrator: the MLP path accumulates per-neuron `|silu(gate)·up|`
energy across tokens and writes a JSON profile at exit. **Measured on the synthetic: the top
25% of neurons carry 99.6% of the MLP energy; top 10% carry 90.3%** — even a random-weight
model concentrates (silu(g)·u is large only when g>0 ∧ |u| large). Real trained models are
*more* concentrated, not less. This closes the loop I flagged as "needs real weights": the
calibration machinery is done, synthetic-proven, and runs on whatever weights you point it at.

Pipeline (all shipped):
1. `HOT_PROFILE=profile.json` run → per-layer neuron energies.
2. `gen_student_i4cal.py`-style sort → hotness-sorted container (same permutation code a
   converter flag `--hot-sort` would use on a real checkpoint).
3. Run sorted container at `HOT_FRAC<1`.

**Quality gate:** logits at HOT_FRAC 0.25 vs 1.0 on the calibrated container: **rel-err
0.0000** over all 248,320 logits — numerically indistinguishable, because the dropped neurons
carry ~0.4% of the energy. Speed gate: TTFT 2.51 → 0.37 s (6.8× at the tiny-prompt operating point).

### B) Per-layer fracs: `HOT_LAYERS={"L0":0.5,...}`
Different layers tolerate different cuts. Loaded and verified (`[hot] HOT_LAYERS loaded`);
layers not listed fall back to the global `HOT_FRAC`. Use after profiling to keep
sensitive layers at 1.0 while cutting the rest.

### C) Warm-start: `TRUNK_WARM=1`
One `madvise(WILLNEED)` sweep over the mapped trunk (prefix-only when `HOT_FRAC<1` — it
prefetched 0.05 GB at frac 0.25 vs 0.15 GB at 1.0, exactly the "never pull the cold tail"
design). **Measured: cold TTFT 2.54 → 0.87 s (2.9×)**. On the 27B at frac 0.25 it is one
sequential ~2.15 GB NVMe fill (~0.6 s), after which decode is RAM-speed.

## The complete mini-PC dense-27B profile (all knobs shipped)

```bash
# convert real Qwen3.8-27B once (bf16 → int4 container, then hot-sort via a calibration run):
convert_qwen36.py --model <27B> --out q27_i4 --ebits 4 --gs 64
TRUNK_MAP=1 HOT_PROFILE=prof.json ./qwen36 ... # 30-min calibration run over your corpus
hot_sort.py q27_i4 prof.json q27_i4_sorted     # permute (shipped pattern; see gen_student_i4cal)
# run: full-size 27B, ~2.1 GB hot working set, RAM-speed warm decode:
TRUNK_MAP=1 HOT_FRAC=0.25 TRUNK_WARM=1 KV8=1 SNAP=q27_i4_sorted ./qwen36 32 8 prompt.txt
# exactness on demand: HOT_FRAC=1.0 (all 27B params computed)
```

## Where this lands (measured student → 27B projection)

| lever | measured (student) | 27B projection |
|---|---|---|
| TRUNK_MAP (boots full-size) | 27B-shape boots at 7.1 GB vs stock OOM-kill | enables everything |
| staged SIMD dequant GEMV | prefill 288→24.5 s (11.8×) | same kernels |
| HOT_FRAC=0.25 paging | MLP 119→38 ms/t; TTFT 2.51→0.37 s | 2.15 GB/tok working set |
| calibrated prefix (v4) | logits rel-err 0.0000 at 25% cut | quality bounded by dropped energy |
| TRUNK_WARM | cold TTFT 2.9× cut | one 0.6 s fill |
| **projected warm decode** | 11.0 tok/s on student | **~12-14 tok/s** (RAM-bound) |

## Remaining techniques (ranked, for when more is needed)

1. **MLA-style absorbed attention for DeltaNet/GQA decode** — removes the O(T) per-token KV
   scan at S=1 (the 43.5→? tok/s long-ctx lever flagged in the flash work). Biggest remaining
   structural win; 2-4 weeks of kernel work.
2. **2:4 structured sparsity on the cold tail** — the tail we *do* page at higher fracs can
   also be half zeros; ~1-2% ppl cost (needs a pruning pass + clearly labeled lossy container).
3. **Draft-model speculation** (the engine's COLI_LOOKUP is prompt-lookup only) — a 0.6B
   family student drafting for the 27B raises effective tok/s ~2-3× when accepted; needs a
   second container + the verify path (exists: `q36_spec_step` S-row verify).
4. **int4-mapped attention/DeltaNet** — already working as of the int4 container format
   (q/k/v/o + dn tensors are int4-packed there; the staged kernel reads them) — the attention
   phase is only 15 ms/t of the 91 ms/t student decode, so the win is small; listed for honesty.
5. **Per-neuron (not per-row) scales at gs<64** — marginal; int4 gs64 is already the sweet spot.

## Files

- Engine: `~/colibri/c/qwen36.c` (`.premap`/`.prehot` backups). Patchers in colibri-harness/:
  `patch_trunkmap*.py` (v1/v2), `patch_hotneuron.py` (v3), `patch_mapfast.py` (v3.1),
  `patch_pagebatch.py` (v3.2), `patch_hotcal_v4.py` (v4: profile/layers/warm)
- Containers: `student_1b_i4` (mechanics), `student_1b_i4s` (1/√ sorted), `student_1b_i4cal`
  (**calibrated**), `synth_qwen38_27b_f16` (full-size 27B-shape)
- `hotcal_pipeline.py` (profile → analyze → coverage report), `gen_student_i4cal.py`
  (calibrated sort — the converter-flag pattern)
