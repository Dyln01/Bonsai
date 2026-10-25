# Neuron window cache (Phase 4b) — design contract

**Date:** 2026-10-20 · status: **CLOSED 2026-10-25 — all three slices done;
gate 3 (field I/O) is a documented NEGATIVE: `Q36_WIN_MB` stays default-off**
(E14; verdict detail under §Gates below).
**Finding (slice 1):** GRIFFIN was dead code engine-wide — `q36_griff_finalize_all`
ran only after generation ended, so no decode or serve turn ever consumed a
Griffin set (this is why Phase 4a measured GRIFFIN == static coverage). Slice 1
finalizes at prefill end; 4a's constant-set negative must be re-run afterwards.
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
2. **CI mechanical (PASSING on `sim` branch):** `Q36_WIN_MB=64` vs `0` on student:
   `DUMP_IDS` identical; `[win]` line present with sane counters.
3. **I/O evidence (mini-PC, HDD):** `tools/run_battery.sh` arms W0/W256/W1G
   (GRIFFIN=1, frac 0.25, window 0/256/1024 MB) record `File system inputs`
   read-bytes + decode ms/tok: window-on must cut read bytes ≥ 50% at
   equal-or-better decode vs W0, else the knob stays default-off and the doc
   says why. **Budget reality from the slice-2 sim:** the window only beats
   static at cap ≥ ~2K neurons; on dense 27B frac 0.25 (K≈4352, ≈3.8 KB/neuron,
   64 layers) that is ≈2.1 GB — infeasible inside the 8 GB envelope, so the
   launcher default stays 0 until the battery says otherwise; the plausible win
   zone is MoE shared experts / smaller-I containers / bigger-RAM boxes.
   **FIELD VERDICT (2026-10-25 overnight matrix,
   `results/battery_llm-030_20261025-133007.json` + `_logs/`; ROADMAP E14):
   FAIL — the knob stays default-off.** Measured: W256 cut fs_read 0.047%
   and W1G 0.018% vs W0 (71.791 / 71.812 vs 71.825 GB) against the ≥50%
   bar; decode was nominally 1.5–1.7% "better" than W0 but inside the ±3%
   seven-arm noise band, so the read gate decides. Why it failed is stronger
   than "no effect": the window NEVER ALLOCATED on q27_i4 — `q36_win_ensure`
   requires int4-packed MLP maps (`map_len == D*I/2`) and this container's
   dense trunk is F16 (`map_len == D*I*2`; `[dense-i8] 0 matrices quantized`
   in every log), so `l->win` stayed NULL and `q36_win_lookup` returned
   before counting: the `[qwen36 win] hits 0 misses 0` line on both window
   arms IS that signature (the same vacuous-pass failure mode slice 1 pinned
   in CI, in the field for a container reason). W0≡W256≡W1G were one
   configuration three times. Independently, the field-forced n_new=2
   (ONE decode step) made ≥50% arithmetically unreachable even on an int4
   trunk: steady-state hits need ≥2 steps, and decode re-read is 17–24 GB
   of the ~71.8 GB total (≤33%). The threshold predates E12/E13's measured
   37-min/token floor; lesson recorded — pre-registered gates must state
   their n_new. What the W arms DID prove: GRIFFIN=1 runs to completion on
   the real container (3/3 rc=0, identical fs_read and greedy token vs the
   non-GRIFFIN arms — on the UNSORTED container the griffin set degenerates
   to the static hot prefix by construction, since observe only sees the
   narrowed Ik = I·frac prefix and top-k re-selects it). **Re-gate
   conditions:** (a) a true int4-trunk container (E13's structural
   follow-up — blocked on the same disk-space wall), (b) n_new ≥ ~8 so
   steady-state hits exist (expected int4 signature at n=2 would have been
   misses ≈ 278k, hits 0), (c) a sorted container (or frac 1.0) so the
   griffin set differs from the prefix; or the slice-2 win zone: MoE shared
   experts / smaller-I containers / bigger-RAM boxes.
4. **Phase 5 arm:** hit-rate + ppl neutrality (values identical by
   construction, so the ppl band is a machinery check, not a quality bet).

## Knobs & report

`Q36_WIN_MB=<n>` — **default 0 everywhere, settled by the gate-3 field
verdict above.** The rotational-media auto-256 originally sketched here was
never shipped into `q36-minipc.sh` and now must not be: on the only real
container the window cannot even allocate (F16 trunk), so an auto-default
would buy a per-layer allocation attempt and a misleading `hits 0 misses 0`
line, nothing else. `Q36_WIN_POLICY=fifo|lru` (fifo default; lru only if the
sim earns it — on synthetic FIFO≈LRU above the cap threshold, no case earned);
report `[qwen36 win] …` beside the existing spec/lookup lines (printed only
when the budget is > 0; counters stay 0/0 whenever allocation was refused —
read that pair as "window inert", not "window useless").

## Implementation slices (one commit each, CI-green each)

1. **DONE:** cache struct + FIFO + gather-path hook + knobs + report line
   (`patch_window.py`, chain hop `.prewin`, 21 hops); CI mechanical gate with
   `GRIFFIN=1` (the gather path runs only on decode rows with a Griffin set —
   the gate initially passed vacuously with hits 0 until that was pinned).
2. **DONE:** `scripts/exp_window_sim.py` + the sim run on existing 4a dumps
   (synthetic curve in `results/results_window_sim_q36.json`; CI replays it
   per push). Budget floor it produced: FIFO beats static only at
   cap ≥ ~2K neurons — ≈2.1 GB on dense 27B frac 0.25, infeasible in 8 GB.
3. **DONE (verdict: FAIL, knob stays 0):** the W0/W256/W1G read-matrix ran
   in the 2026-10-25 field overnight
   (`results/battery_llm-030_20261025-133007.json`): read-byte cuts
   0.047%/0.018% vs the ≥50% bar, and the window never allocated (F16
   trunk vs the int4-only guard) — full verdict + re-gate conditions under
   §Gates gate 3 and in ROADMAP E14.

## Out of scope

Predictors (4a closed on synthetic, re-gates in Phase 5), cross-engine KV8/TQ
ports (Phase 4c, separate contract — sibling engines are not in this repo or
its CI), MoE expert cache changes (it already is a window).
