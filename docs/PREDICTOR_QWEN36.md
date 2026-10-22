# Deja Vu-class predictor + Apple window (Phase 4) — design contract

**Date:** 2026-10-20 · status: **offline gate RUN — documented NEGATIVE on synthetic (machinery retained for Phase 5)**
**Grounding:** `docs/RESEARCH_MAP_DENSE.md` (Deja Vu / PowerInfer-dynamic /
LLM-in-a-Flash windowing / ShadowLLM rows) · **Predecessors:** HOT_FRAC
(static calibrated prefix), GRIFFIN v5 (per-prompt selection, training-free),
TRUNK_MAP (mmap trunk).

## The one architecture, two names

The literature's two remaining pieces are a single scheme on our stack:

- **Predictor** (Deja Vu/ShadowLLM): per-TOKEN active-neuron set predicted
  from the layer input. Their reported regime: ~95% of MLP skippable
  per token, a 4-10%-width 2-layer predictor, computed ahead of the MLP.
- **Windowing** (LLM in a Flash): a small resident cache of the RECENT
  predicted sets, so per-token I/O only touches set drift, not the whole
  set — reported DRAM fraction 10% -> 2.4%.

Without per-token sets there is nothing to window (our GRIFFIN set is
per-SEQUENCE and already resident-able once); so the predictor is the
gate to the whole item. Per the ROADMAP rule: **the predictor must beat
GRIFFIN or the item stays a documented negative** — and that verdict is
measurable OFFLINE, before any engine runtime work:

## The offline gate (G2) — no runtime work until it passes

Data: a new profiling mode dumps, per (layer, token), the pair the
predictor needs:

- input `x` = the layer's MLP input row (post post_ln, D floats), and
- the TRUE per-token top-K neurons with their energies
  (|silu(g)*u|, K=64), taken from a FULL compute (HOT_FRAC=1, GRIFFIN=0).

Engine knob `Q36_PROFILE_TOK=<path prefix>` -> one raw binary per layer
(`<prefix>_L<layer>.bin`, record = `x[D] f32 | K u32 idx | K f32 energy`).
The hook sits beside the existing `q36_hot_acc_add` call sites (both the
S=1 and the batched prefill paths), so a plain corpus prompt captures
prefill and decode rows alike.

Trainer (numpy, harness-side, no engine change): per layer, logistic
regression `D -> I` (linear first; add the hidden layer only if the
linear one fails the gate) fit on corpus A's dumps with class-weighted
BCE. Training uses ONLY the same container's own activations — the same
legality class as the existing HOT_PROFILE calibration loop (nothing is
downloaded; this is self-calibration on synthetic containers).

**Gate G2 (the decision):** on a DISJOINT corpus B (drifted), per layer
per token, energy coverage = sum of true-top-K energies that the selected
set also contains / sum of true-top-K energies. Compare three selectors:

1. **static-A** — top-K by corpus-A aggregate energy (the HOT_FRAC
   model),
2. **GRIFFIN-B** — top-K of corpus B's own prompt aggregate (per-prompt
   selection, the incumbent),
3. **predictor-B** — the fitted model's per-token top-K.

**Decision rule (pre-registered): the predictor's mean energy coverage
on corpus B must beat BOTH baselines by >= 5 percentage points.** If it
does, the runtime work unlocks (per-token gather through the GRIFFIN
machinery + the resident window over recent sets). If it does not, the
item is a documented negative with the measured margins, and the reason
is recorded — the expected negative mode on random weights is that the
per-token contextual tail is small relative to the static power-law core
(real Deja Vu numbers live on real weights, which is Phase 5's premise).

## Remaining gates (pre-registered, only after G2 passes)

- **G1 mechanical:** predictor-off = stock bit-identity; window-off
  separately. Runtime gather composes with GRIFFIN's row-view QW +
  column-gather down unchanged.
- **G3 quality (Phase 5 arm):** wikitext-2 ppl with predictor+window vs
  full, band-style (protocol doc).
- **G4 speed:** meaningful only in the streaming regime (mapped
  full-size container, cold cache): resident-window hit rate + honest
  decode ms/token vs prefix-only. On RAM-sized synthetics the speed arm
  is directional only.
- **Window definition (runtime):** per layer, the union of the last C
  tokens' predicted sets (C default 8), re-madvise(WILLNEED)'d on drift;
  the union IS Apple's window — DRAM cost ~ C*K rows.

## Out of scope

Cross-engine KV8/TQ ports (separate Phase 4 item, mechanical), learned
predictors on real weights (Phase 5), ReLUfication routes (house rule).

## The offline gate ran (2026-10-20) — negative, with the sharpest possible reason

Machinery shipped: `Q36_PROFILE_TOK` per-token dumps (`patch_profile_tok.py`,
backup `.preptk`, chain 17 hops) + `scripts/exp_predictor_q36.py` (loader,
ridge fit, three-way coverage gate). Corpora: disjoint 4096-token halves of
the long prompt, `student_1b_i4cal`, full compute, K=64, all runs rc=0.

| selector | mean energy coverage over 20 layers |
|---|---|
| static-A (train-corpus aggregate) | **1.0000** |
| static-B (eval-corpus aggregate) | **1.0000** |
| GRIFFIN-j (per-sequence prompt selection) | **1.0000** |
| predictor (per-token ridge fit) | 0.9996 |

**The per-token true top-K set is CONSTANT on this container**: the
aggregate set covers 100.00% of every token's top-K energy on every layer.
On random weights the energy ranking is norm-driven and content-independent
— the Deja Vu premise (per-token contextual variation) is absent *by
construction*. Not "the tail is small" — **the tail does not move**. There
is nothing for a predictor to add and nothing to window; the window item
dies with the predictor on this container class. GRIFFIN remains the
strongest selector available, and it cannot be beaten here because every
selector is exactly the true set.

Per the pre-registered decision rule the runtime work (per-token gather +
resident window) is **not unlocked**. The machinery is exactly what Phase 5
needs: the same dump mode on a REAL container measures the true per-token
variation; if real weights show moving top-K sets, the same fit + coverage
harness runs unchanged and the +5pp rule decides.

**Harness lesson (recorded in results_predictor_q36.json):** the dump
record mixes f32 and u32; loading it as one float array and *casting* the
index slice turns every u32 into a denormal 0 — the first G2 run produced a
perfect four-way 0.039 "tie" that was pure loader artifact. A structured
dtype (`x f32×D | idx u32×K | en f32×K`) is mandatory. A gate
that returns identical values for every arm is a loader bug until proven
otherwise.


## Re-run with live GRIFFIN (2026-10-22, `results/results_predictor_rerun_q36.json`)

After Phase 4b slice 1 revived dense GRIFFIN (finalize at prefill end), G2 was
re-run on disjoint synthetic corpora: **all four selectors score 1.0000** — the
decision rule FAILs again, but the diagnosis changes: coverage *saturates* on
synthetic weights (energy mass concentrates in a few norm-dominant rows), while
per-token sets demonstrably vary (window sim: jaccard(t,t-1)=0.17). The original
"sets are constant" finding was thus twice an artifact: inert GRIFFIN first,
metric saturation second. G2 can only discriminate on real weights; the Phase 5
arm stands as designed. Predictor remains offline-only.
