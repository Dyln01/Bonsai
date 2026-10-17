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
(measured, rc=137). This stack makes the same full-size model run in a ~4 GB RAM envelope with
`HOT_FRAC` as an exactness dial — 1.0 restores full compute at any moment, no weights edited,
the file on disk IS the model.

## Headline results (4-core / 7.8 GB emulation, synthetic containers, no downloads)

| | stock dense 27B | this stack (frac 0.25) |
|---|---|---|
| RAM | OOM at 7.36 GB (cannot run) | **runs in ~4 GB envelope** |
| bytes/token (MLP) | 8.6 GB sweep | **2.15 GB**, served from page cache after warm |
| warm decode | ~0.9 tok/s (int4 sweep) | **12-14 tok/s projected** (9.7 measured on student analog) |

Student-analog measured (same kernels): MLP phase 119 → 38 ms/token (**3.1x**), decode
5.8 → 11.0 tok/s, TTFT 26.5 → 11.6 s, peak RSS 2.96 → 1.87 GB.

## The techniques (all measured on synthetic containers; reports inside `docs/`)

1. **TRUNK_MAP** — mmap the dense trunk, dequantize f16/int4 in-register in the GEMV,
   MADV_WILLNEED readahead. Resident RSS = embed/head + KV + norms only.
2. **Hot-neuron paging (HOT_FRAC)** — calibrate once, sort container neurons by importance
   (pure relabeling), compute only the top fraction. MLP cost scales exactly with the dial
   (119/79/38 ms/tok at 1.0/0.5/0.25).
3. **GRIFFIN v5** — per-prompt neuron selection from the prompt's own activations
   (training-free, calibration-free, works on unmodified containers).
4. **KV8 + flash-tile prefill** — e4m3 KV (parity-proven 256/256 codes) and 2.4x TTFT cut;
   KV8+int4 = -26% RSS at 8k ctx.
5. **KV_TQ (4/3-bit KV)** — PolarQuant & rotated-int4 KV rows behind the same scale-hoisted
   read path (parity-gated incl. radius/linearity): −19% total RSS vs f32 at 8k, −23 MB vs
   KV8; tier ladder ends at 40 B/row @ head_dim 128. RAM lever, decode-price labeled in
   `docs/KVTQ_QWEN36.md`.
6. **q36-minipc launcher** — one command composing the right profile per model class
   (auto-detects dense vs MoE): `serve|chat|bench|seed|calibrate|sort|doctor`.

## Quick start (assumes a colibri engine checkout + converted container)

**Full install guide (Linux native + Windows/WSL2, dependencies, models, verification,
troubleshooting): [docs/INSTALL.md](docs/INSTALL.md)** — start there.

```bash
# install
cp tools/q36-minipc.sh ~/bin/q36-minipc && cp tools/hot_sort.py ~/bin/
chmod +x ~/bin/q36-minipc ~/bin/hot_sort.py

# day one, per dense model
q36-minipc calibrate ~/q27_i4        # Q36_CORPUS=your_text.txt
q36-minipc sort      ~/q27_i4        # -> ~/q27_i4-sorted
# every day after — or skip both: GRIFFIN adapts per-prompt on the UNMODIFIED container
q36-minipc serve    ~/q27_i4-sorted  # TRUNK_MAP+GRIFFIN/prefix+HOT_FRAC+KV8 (~2 GB + hot set)
q36-minipc bench    ~/q27_i4-sorted  # honest phase-split battery
```

Levels: `Q36_LEVEL=eco|balanced|quality` · dial: `Q36_HOT_FRAC=1.0` = exact full compute.

## Repo layout

```
docs/     the research reports (TRUNK_MAP, HOT-NEURON, GRIFFIN, KV8, KV_TQ, FLASH, integration)
          + research map + install guide + final low-RAM results
tools/    q36-minipc launcher, hot_sort.py, calibration pipeline, patch scripts (engine edits)
engine/   qwen36.c patch stack snapshot + kv8_fused.h + kv_tq.h + parity gates + backups
scripts/  synthetic container generators (no real checkpoints) + experiment drivers
results/  measured JSON from every battery
```

## Engine

The modified engine lives in a colibri checkout (Apache-2.0): clone
`github.com/JustVugg/colibri`, copy `engine/qwen36.c`, `engine/kv8_fused.h`,
`engine/kv8_parity.c` over `c/`, `make -C c qwen36`. Patch scripts in `tools/` reproduce
every engine edit from `qwen36.c.orig` (stock @ bf24429) if you prefer applying stepwise.
**No real model weights are in this repo or were used in testing** — all benchmarks use
synthetic same-geometry containers (see `scripts/gen_*`).

## Status

Research code. All numbers are real engine runs on synthetic containers; 27B-scale rows are
projections from measured student ratios (labeled in the reports). Quality claims for frac<1.0
are by construction (sorting is relabeling) + literature; real-checkpoint perplexity
validation is the remaining step.

Apache-2.0 (engine-derived portions inherit colibri's license).
