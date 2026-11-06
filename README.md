<div align="center">
<img src="./logo.png" width=180 height=180 />   
   
# Bonsai

### **Build on colibri engine**

**Run full-size dense 27B-class LLMs (and MoE siblings) on 8 GB mini-PCs — no GPU, no downsizing.**
Mmap'd trunk + hot-neuron paging + per-prompt neuron selection (GRIFFIN) + tiered KV cache
(fp8 → polar/int4) + flash-tile prefill + self-speculative decoding, composed by a single one-knob launcher.

Built as a research patch stack on the open-source **colibri** inference engine (Apache-2.0,
not included — see *Engine* below).

*formerly `qwen36-minipc` — the launcher command stays `q36-minipc`*

</div>

## Why this exists

A dense 27B needs ~22 GB of resident weights; the stock engine is OOM-killed on an 8 GB box
(measured, rc=137). This stack makes the same full-size model **run** on a 7.62 GB mini-PC —
field-measured at **6.65–7.01 GB peak RSS**, full compute, rc=0 (`HOT_FRAC=1.0` restores full
compute at any moment, no weights edited, the file on disk IS the model). The `HOT_FRAC`
exactness dial below 1.0 requires a hotness-**sorted** container; on the unsorted field
container it is a documented negative (ppl 6.74 → 1791 at frac 0.25) and the sorted-container
arm is pending — see the Phase 5 report.

## Headline results (REAL WEIGHTS — Qwen3.8-27B-class dense, llm-030: 4-core / 7.62 GB / HDD)

Phase 5 W1 session, completed 2026-10-30 — full report:
**[docs/REALWEIGHTS_QWEN36.md](docs/REALWEIGHTS_QWEN36.md)** (tag `phase-5`; every number
below traces to `results/realweights_20261026-1014/`).

| | measured |
|---|---|
| runs at all | **yes** — stock OOMs (rc=137); this stack ran every arm rc=0 |
| peak RSS | **6.65–7.01 GB** of 7.62 GB (M4 PASS ≤ 7.8 GB) |
| quality, full compute (frac 1.0) | **tf-NLL 1.9076 nats/tok, ppl 6.74** (wikitext-2-raw, 512 ctx / 64 scored) |
| quality, frac 0.5 / 0.25, UNSORTED container | ppl 29.24 / 1791.28 — **documented negatives**: the dial's contract is a hotness-sorted prefix; sorted arm E pending on disk space |
| determinism (M1) | **PASS** — 576-id greedy streams byte-identical ×2 at frac 1.0 |
| speculative decoding (first live numbers) | **82.0% acceptance, 2.86 tokens/forward** |
| TTFT (512 tok @ frac 1.0) | 2460.5 s |
| decode | **~2000–2350 s/token** (`[timers] TOTAL` frac 0.25; spec-gate T(1) frac 1.0) — HDD streaming floor: the F16 trunk's frac-0.25 shared-MLP prefix alone (~8.6 GB) exceeds the box's entire 7.62 GB RAM, so nothing stays warm |

The old synthetic-container projection of "12–14 tok/s warm" is **de-projected** by the
measured row above: it was arithmetic for an int4 trunk whose frac-0.25 hot prefix (~2 GB)
fits the page cache; this checkpoint's container is F16 (54.7 GB trunk — its frac-0.25
shared-MLP prefix alone, ~8.6 GB, exceeds the box's entire 7.62 GB RAM). An int4-packed
container is the measured path to the warm regime (§8 of the report).

## The numbers are a function of the host now, not a slogan

**`docs/COST_MODEL_QWEN36.md`** — the de-projection's replacement. Two shipped tools:

* **`tools/pack_trunk_i4.py`** — offline dense-trunk int4 packer (disk→disk, streaming,
  no engine change; the engine's map path dispatches on `dtype == U8 && nbytes == I*O/2`).
  Packs `lm_head`, the MLP, the router, attention q/k/v/o and the DeltaNet projections to
  the engine's mapped-int4 dialect with per-row `.qs` scales; `--profile` writes the
  hotness-sorted prefix **in the same pass** (what `HOT_FRAC`'s contract needs).

  | | F16 (field container) | packed |
  |---|---|---|
  | dense trunk bytes | 1× | **0.25×** (measured 3.93× on 51 tensors) |
  | MLP prefix at frac 0.25 | 8.56 GB | **2.14 GB** |
  | fits the box's RAM at frac 0.25? | no (needs frac ≤ 0.142) | **yes (fits at frac ≤ 0.792)** |
  | predicted decode, same frac | 2,384 s/token | **99 s/token** |

* **`tools/project_q36.py`** — `q36-minipc project`-class cost model: per-class byte
  attribution from the container's own shard headers, cold and warm columns, the frac at
  which the working set fits RAM, and a `--probe` that measures *this* host's read rate.
  `--field-check` re-derives the recorded field points and fails if the model drifts out of
  band (currently +7.1% on decode, −15.7% on TTFT; the frac-1.0/512 TTFT outlier is
  reported, not tuned away).

The attribution is the part the old README could not show: at frac 0.25 the MLP dial
governs **27%** of the per-token bytes — the other 73% (DeltaNet 57%, lm_head 8%,
attention 7%) has no dial, no paging and no quantization on the mapped path. No kernel
work closes that; only fewer bytes and a medium fast enough to move them.

Also shipped, each with its code *and* its number:

* **`tools/make_layer_fracs.py`** — per-layer `HOT_FRAC` schedule: layers are not equally
  compressible and every layer costs the same bytes per neuron at int4, so the resident
  budget is a knapsack over per-layer marginal energies (Lagrangian relaxation), emitted in
  the flat `"L<n>": <frac>` form the engine's naive `HOT_LAYERS` scan actually reads and
  verified against a re-implementation of that scan. **+81% captured energy vs a uniform
  fraction at equal bytes** on a heterogeneous profile. Picked up automatically by the
  launcher (`<model>/hot_layers.json`).
* **The window cache, now that it can allocate** — Phase 4b was blocked only by the F16
  trunk; the packer unblocks it. Priced with the engine's own capacity formula (512 MB =
  1090 rows/layer = 25.1% of the frac-0.25 active set). The launcher auto-enables it on an
  int4 trunk, and prints the reason + the fix when the trunk is F16.

`project_q36.py --stack` computes the whole ladder on this host — every row model output:

| stack | bytes/tok | s/token | what changed |
|---|---|---|---|
| F16 trunk | 33.3 GB | **2536** | the container as shipped |
| int4 trunk | 8.3 GB | **138** | `pack_trunk_i4.py` |
| + layer schedule | 7.8 GB | **92** | `make_layer_fracs.py`, equal bytes |
| + window cache | 8.3 GB | **90** | `Q36_WIN_MB` (marginal once the trunk is int4 — see the report) |
| + speculation | 8.3 GB | **31** | MTP, *measured* 2.86 accepted/forward |

Gates: `tools/test_pack_trunk_i4.py` (layout oracle, `pack∘sort == sort∘pack`
byte-for-byte over 102 tensors, predicates, refusals), `tools/test_layer_fracs.py`
(engine-readable format, budget, energy capture) and `scripts/check_projection_field.py`
(field fit + the window's capacity against the engine's formula) — all in CI, plus an
engine-level parity arm in `engine-build.yml`.

## Earlier results (4-core / 7.8 GB emulation, synthetic containers, no downloads)

| | stock dense 27B | this stack (frac 0.25) |
|---|---|---|
| RAM | OOM at 7.36 GB (cannot run) | **runs in ~4 GB envelope** (synthetic student shape; field-measured on the real 27B: 6.65–7.01 GB — table above) |
| bytes/token (MLP) | 8.6 GB sweep | **2.15 GB**, served from page cache after warm |
| warm decode | ~0.9 tok/s (int4 sweep) | ~~12-14 tok/s projected~~ **de-projected by Phase 5: ~2000+ s/token measured on the F16 field container** (projection assumed an int4-sized hot prefix; 9.7 tok/s stands only for the RAM-resident student analog) |

Student-analog measured (same kernels): MLP phase 119 → 38 ms/token (**3.1x**), decode
5.8 → 11.0 tok/s, TTFT 26.5 → 11.6 s, peak RSS 2.96 → 1.87 GB.

## The techniques (all measured on synthetic containers; reports inside `docs/`)

1. **TRUNK_MAP** — mmap the dense trunk, dequantize f16/int4 in-register in the GEMV,
   MADV_WILLNEED readahead. Resident RSS = embed/head + KV + norms only.
2. **Hot-neuron paging (HOT_FRAC)** — calibrate once, sort container neurons by importance
   (pure relabeling), compute only the top fraction. MLP cost scales exactly with the dial
   (119/79/38 ms/tok at 1.0/0.5/0.25). **Field verdict (Phase 5): the quality mechanism
   requires the sorted container — on the unsorted field container frac<1.0 is random neuron
   dropout (ppl 6.74 → 1791.28, documented negative); sorted arm E pending**
   (`docs/REALWEIGHTS_QWEN36.md` §3).
3. **GRIFFIN v5** — per-prompt neuron selection from the prompt's own activations
   (training-free, calibration-free, works on unmodified containers).
4. **KV8 + flash-tile prefill** — e4m3 KV (parity-proven 256/256 codes) and 2.4x TTFT cut;
   KV8+int4 = -26% RSS at 8k ctx.
5. **KV_TQ (4/3-bit KV)** — PolarQuant & rotated-int4 KV rows behind the same scale-hoisted
   read path (parity-gated incl. radius/linearity): −19% total RSS vs f32 at 8k, −23 MB vs
   KV8; tier ladder ends at 40 B/row @ head_dim 128. RAM lever, decode-price labeled in
   `docs/KVTQ_QWEN36.md`.
6. **KV-prefix reuse** — serve-mux transcript reuse across agent turns: 93% of a repeated
   prompt served from standing KV (10-12x faster turns 2+), one slot per conversation
   (`Q36_SLOTS`); agent contract + measured rules in `docs/PREFIX_REUSE_QWEN36.md`.
7. **Q36_FAST_SOFTMAX** — AVX2 fast exponential in the attention softmax: 9.5-11.5x the
   kernel, −6.5% attention ms/token at 16k ctx, TTFT-neutral and *more accurate than stock*
   (f32 accumulator drift); opt-in, two negative results documented alongside
   (`docs/SOFTMAX_QWEN36.md`).
8. **GQA-absorbed decode tiles (Phase 3a)** — dequant each KV tile once per GQA group
   instead of once per query head: KV8 −34.5%, KV_TQ4 −47% attention ms/token at 16k,
   KV8 −27.4% at the 68.4k fixture; outputs bit-identical (A/B binary DUMP_IDS gates),
   RAM unchanged, no knobs (`docs/GQA_ABSORB_QWEN36.md`).
9. **q36-minipc launcher** — one command composing the right profile per model class
   (auto-detects dense vs MoE): `serve|chat|bench|seed|calibrate|sort|doctor`.
10. **Q36_SCAN two-tier decode scan (Phase 3b-scan)** — window + measured sinks +
    salience-gated 128-row blocks; uint64 fixed-point salience keeps the scan set
    bit-reproducible under OpenMP; default OFF (`Q36_SCAN=full` preserves stock
    bit-exactly); machinery ceiling 3.2% rows scanned -> 11.9x attention at 16k
    (`docs/SCAN_DECAY_QWEN36.md`).

Landed on `test` (CI-gated, pending merge + real-hardware battery): **MTP self-speculative
head** — loader, dense+MoE drafting/verify on the existing spec machinery, force-oracle
proofs (forced-accept reproduces the oracle sequence exactly; rejections stay exact),
resident int4 experts — `docs/MTP_DESIGN_QWEN36.md`, phase board in `docs/ROADMAP.md`.

## Quick start (assumes a colibri engine checkout + converted container)

**Full install guide (Linux native + Windows/WSL2, dependencies, models, verification,
troubleshooting): [docs/INSTALL.md](docs/INSTALL.md)** — start there.
**Every command, knob and log line, copy-paste ready: [docs/COMMANDS.md](docs/COMMANDS.md)**.

```bash
# install (the launcher resolves these tools from ~/bin or from the repo checkout)
cp tools/q36-minipc.sh ~/bin/q36-minipc
cp tools/hot_sort.py tools/pack_trunk_i4.py tools/make_layer_fracs.py tools/project_q36.py ~/bin/
chmod +x ~/bin/q36-minipc

# day one, per dense model
q36-minipc calibrate ~/q27_i4        # Q36_CORPUS=your_text.txt
q36-minipc pack      ~/q27_i4        # -> ~/q27_i4-i4  (int4 trunk + sorted prefix; ~1/4 the disk)
q36-minipc fracs     ~/q27_i4-i4     # -> hot_layers.json (per-layer HOT_FRAC, auto-used)
q36-minipc project   ~/q27_i4-i4     # what it will cost on THIS host, and the stack ladder
q36-minipc bench     ~/q27_i4-i4     # window cache auto-on for an int4 trunk
# optional, skip if you packed with --profile above:
q36-minipc sort      ~/q27_i4        # -> ~/q27_i4-sorted
# every day after — or skip both: GRIFFIN adapts per-prompt on the UNMODIFIED container
q36-minipc serve    ~/q27_i4-sorted  # TRUNK_MAP+GRIFFIN/prefix+HOT_FRAC+KV8 (~2 GB + hot set)
q36-minipc bench    ~/q27_i4-sorted  # honest phase-split battery
```

Levels: `Q36_LEVEL=eco|balanced|quality` · dial: `Q36_HOT_FRAC=1.0` = exact full compute.

## Repo layout

```
docs/     the research reports (TRUNK_MAP, HOT-NEURON, GRIFFIN, KV8, KV_TQ, PREFIX-REUSE,
          FLASH, integration) + research map + install guide + command reference
          + dense-27B benchmark walkthrough + final low-RAM results
tools/    q36-minipc launcher, hot_sort.py, calibration pipeline, patch scripts (engine edits)
engine/   qwen36.c patch stack snapshot + kv8_fused.h + kv_tq.h + parity gates + backups
scripts/  synthetic container generators (no real checkpoints) + experiment drivers
results/  measured JSON from every battery
```

## Engine

The modified engine lives in a colibri checkout (Apache-2.0): clone
`github.com/JustVugg/colibri`, copy `engine/*.c` and `engine/*.h` over `c/`
(the glob on purpose: the header set grows with phases — an explicit list
went stale once already), `make -C c qwen36`. The `engine/` snapshots are the
source of truth (md5-pinned in `tools/patch_manifest.json`), and the patch
scripts in `tools/` reproduce them **byte-exactly from stock** — the full chain
orig→premap→prehot→pretq→presm→final replays under `tools/verify_engine.sh`
and in CI on every push (E1 drift audit closed 2026-10-19; superseded
historical patchers in `tools/attic/`).
**No real model weights are in this repo, ever** — CI and all sandbox benchmarks use synthetic
same-geometry containers (see `scripts/gen_*`). The one real-checkpoint session (Phase 5, W1)
ran exclusively on the mini-PC outside the repo, under the recorded sign-off
(`docs/VALIDATION_PROTOCOL_QWEN36.md` §6); only logs, hashes and metrics were committed
(`results/realweights_20261026-1014/`).

## Status

Research code. Synthetic-container numbers are real engine runs (labeled in the reports);
27B-scale synthetic rows are projections from measured student ratios. **Real-checkpoint
validation (Phase 5, W1) is COMPLETE on real weights** (2026-10-30, tag `phase-5`):
full-compute dense 27B ran end-to-end on the 7.62 GB mini-PC at ppl 6.74
(wikitext-2-raw, 512/64), M1-deterministic, RSS ≤ 7.01 GB — with the frac-dial, KV8 and
KV_TQ4-absorb arms recorded as pre-registered documented negatives and the sorted-container
arm pending disk space: **[docs/REALWEIGHTS_QWEN36.md](docs/REALWEIGHTS_QWEN36.md)**.
W2 (MoE + MTP) is the follow-up session.

Apache-2.0 (engine-derived portions inherit colibri's license).
