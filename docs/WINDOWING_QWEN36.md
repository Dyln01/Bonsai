# Neuron window cache (Phase 4b) — design contract

**Date:** 2026-10-20 · status: **slice 1 shipped (2026-10-22, `main`)** — slices 2-3 open
**Predecessor:** Phase 4a (`docs/PREDICTOR_QWEN36.md`) measured the per-token
top-K neuron sets on synthetic weights and found them **constant** (norm-driven,
content-independent): the Deja-Vu predictor premise is absent there by
construction, and so is windowing's *synthetic* value case. This contract is
written around that finding, not despite it.

## Where windowing can and cannot prove itself

| regime | per-token active set | window's effect |
|---|---|---|
| synthetic, static frac | constant (4a) | none — degenerates to the static hot prefix |
| synthetic, GRIFFIN | constant per prompt, changes per prompt | none within a turn; warm-cache across turns only |
| **real weights** | varies per token with temporal locality (Apple LLM-in-Flash: windowing cut DRAM 10%→2.4%) | **the value case** — Phase 5 arm |
| **any weights, GRIFFIN on HDD** | scattered row indices re-read from disk per token | **the value case now**: resident cache turns seeks into RAM hits |

So: the mechanism ships gated on *mechanical* and *I/O* evidence; its *quality*
case waits for real weights exactly like 4a's predictor arm does.

## Mechanism

Per-layer resident row cache ("the window") over the dense MLP's gather path
(`moe_ex`'s `gs->idx` loop, the shared-expert/dense branch):

- entry = one neuron row triplet (gate/up/down packed int4 bytes, copied
  byte-identical from the mapped trunk) + its index;
- policy FIFO, capacity `Q36_WIN_MB` (default 256 MB total, split evenly over
  MLP-bearing layers; 0 = off);
- per token with active set A: hit → gemv from the resident copy; miss → gemv
  from the map (today's path) + insert, evicting FIFO;
- **determinism:** the packed copy is byte-identical to the map row, so every
  gemv sees the same bits whichever source answers — output is independent of
  cache history, and CI id-identity gates apply unconditionally;
- telemetry: `[win] hit-rate p50/p90 per layer, inserts, evicts` report line +
  counters in `RouteStats`-style struct for the battery.

Relationship to existing machinery: the MoE **expert LRU cache is this idea
already** for expert slabs; the window extends it to dense neuron rows.
`sort`/hot-prefix makes hot rows contiguous (sequential reads); the window
makes *active* rows resident (no reads at all). They compose: sorted container
+ window = contiguous miss stream + RAM hit path.

## Gates, in order (any fail = documented negative, knob stays 0)

1. **Offline simulation first** (`scripts/exp_window_sim.py` on 4a's
   `Q36_PROFILE_TOK` dumps): hit-rate vs capacity curve for FIFO/LRU on the
   measured per-token sets. On synthetic this validates bookkeeping only
   (constant sets ⇒ 100% after warm) — the curve that matters runs on real
   weights in Phase 5 with the same script.
2. **CI mechanical:** `Q36_WIN_MB=256` vs `0` on student + synth_qwen27:
   `DUMP_IDS` identical; `[win]` line present with sane counters.
3. **I/O evidence (mini-PC, HDD):** `exp_window_q36.py` reads
   `/proc/<pid>/io` read-bytes and decode ms/tok for GRIFFIN on/off × window
   on/off at frac 0.25: window-on must cut read bytes ≥ 50% at equal-or-better
   decode vs window-off GRIFFIN, else the knob defaults off and the doc says why.
4. **Phase 5 arm:** hit-rate + ppl neutrality (values identical by
   construction, so the ppl band is a machinery check, not a quality bet).

## Knobs & report

`Q36_WIN_MB=<n>` (default 256 when GRIFFIN=1 on rotational media, else 0 —
the launcher probes `/sys/block/*/queue/rotational` like the calibrate ETA
does); `Q36_WIN_POLICY=fifo|lru` (fifo default; lru only if the sim earns it);
report `[qwen36 win] …` beside the existing spec/lookup lines.

## Implementation slices (one commit each, CI-green each)

1. **DONE:** cache struct + FIFO + gather-path hook + knobs + report line
   (`patch_window.py`, chain hop `.prewin`, 21 hops); CI mechanical gate with
   `GRIFFIN=1` (the gather path runs only on decode rows with a Griffin set —
   the gate initially passed vacuously with hits 0 until that was pinned).
2. `scripts/exp_window_sim.py` + the sim run on existing 4a dumps (synthetic
   curve into `results/results_window_sim_q36.json`).
3. `scripts/exp_window_q36.py` I/O gate + battery arms for the mini-PC
   (window × GRIFFIN matrix), numbers into the 2b-4 battery doc.

## Out of scope

Predictors (4a closed on synthetic, re-gates in Phase 5), cross-engine KV8/TQ
ports (Phase 4c, separate contract — sibling engines are not in this repo or
its CI), MoE expert cache changes (it already is a window).
