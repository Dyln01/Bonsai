# Decode scan decay (Phase 3b-scan) — design contract

**Date:** 2026-10-20 · status: **implemented + machinery-gated (shipped, knob default `full`)**
**Predecessors:** 3a (GQA-absorbed tiles) removed per-head dequant redundancy;
3b-rot/3b-prefill removed the TQ per-row/per-block FWHT. What remains in decode
attention is the **row count itself**: every (kv-head, row) pair scans all
`qpos+1` KV rows. Measured post-3a: **57.65 ms/tok at the 68.4k fixture**
(`docs/GQA_ABSORB_QWEN36.md`); that is O(T) per token and grows linearly —
the only structural speed lever left below the model math itself.

## Why not the obvious things

- **KV tiers don't help:** KV8/TQ shrink bytes per row, not rows scanned.
- **Full MLA absorption** (FlashMLA math, fold q·K into weight space) needs a
  latent-KV architecture; qwen36's GQA stores per-head K rows — there is no
  latent to absorb into. 3b-rot absorbed what GQA + a fixed rotation allows.
- **Learned predictors** (Deja Vu/ShadowLLM class) violate the training-free
  house rule and belong to Phase 4's dynamic-predictor item, not here.

## The contract: two-tier deterministic scan

Tier 1 — **always scanned:** a local window `W` (default 4096 rows) plus
**sink blocks** (first 256 rows per layer; attention-mass concentration at the
sequence start is the one universal prior this repo is willing to hard-code,
and it is measurable per layer in the salience pass below — if a container
shows no sink concentration, the sink tier degrades to 0 rows by measurement,
not by faith).

Tier 2 — **salience-gated blocks beyond W:** per (layer, kv-head) block table
over 128-row blocks. A block's salience is the running sum of attention mass
its rows received, accumulated **where scores already exist** (flash prefill
inner loop; decode score loop) — zero extra passes, one f32 add per (row, head)
into `sal[layer][kvh][block]`, normalised per head at use time.

Decode scan set for (layer, kvh, head-group) = sinks ∪ window ∪ {blocks in
descending salience until cumulative share ≥ 1−ε}, ε default 0.05. The set is
a deterministic function of (salience state, qpos, knobs) — no sampling, no
learning, so reruns are bit-reproducible for fixed knobs.

## Knobs (defaults preserve today's behavior)

`Q36_SCAN=full|two-tier` (default **full** until §Gates pass on real weights) ·
`Q36_SCAN_WIN=4096` · `Q36_SCAN_EPS=0.05` · `Q36_SCAN_SINK=256`.
Report line `[qwen36 scan] layer-avg scanned rows / total, mass kept`.

## Correctness interactions (the parts that bite)

1. **Speculative verify / rollback:** the salience table is model state that
   evolves per token. `q36_spec_*` snapshots must include it (alongside DN
   rec/conv in the snap slots) or a rejected draft leaves salience from rows
   that never stood — nondeterminism. Rollback = slot swap, same pattern.
2. **KV-prefix reuse:** a reused prefix's salience must come from the stored
   state, not be re-accumulated — pin/restore path carries it with the DN state.
3. **KV8/TQ staging:** sparsity *reduces* dequant work (only scanned rows are
   dequantized) — compose with 3a/3b by building the tile list from the scan
   set instead of `t0..tmax`.
4. **DeltaNet layers:** untouched (no KV scan).
5. **Causal mask + t-ascending V accumulation:** unchanged within the scanned
   set; skipped rows contribute exactly 0 mass by construction of the softmax
   renormalisation over the scanned set (documented approximation, hence §Gates).

## Gates (in order; any fail = documented negative, knob stays `full`)

1. **Mechanical (CI, synthetic):** `Q36_SCAN=full` id-identity with today's
   binary (already true — the flag must be a no-op path); two-tier runs
   rc=0 + deterministic (two runs identical `DUMP_IDS`).
2. **Salience self-check (CI, synthetic):** on a repeated-ngram prompt, the
   ngram's blocks rank in the scan set (the mechanism tracks mass it sees).
3. **Quality (Phase 5 arm, real weights):** wikitext-2 ppl of arm G with
   `Q36_SCAN=two-tier` vs `full` ≤ **+0.5% rel** (protocol band style;
   pre-registered here, measured there).
4. **Speed (mini-PC battery):** 68.4k fixture decode ms/tok ≥ 2× faster at
   ε=0.05 with scanned-rows report ≤ 25% of total; else the lever is labeled
   dead for this architecture and the doc records why.

## Implementation sketch (for the patcher, not a promise)

`sal` f32 `[n_layers][kv_heads][max_t/128]` resident (~64·8·512·4 B = 1 MB at
68k — noise); flash inner loop adds `p` (post-softmax weight, pre-V) per block;
decode adds the score-row mass after `softmax_row`; scan-set builder returns an
int list per (layer, kvh) cached per token (salience changes only via the
current row's own mass, which is < ε of the budget by construction — recomputed
every token anyway, it is one argsort over ≤ max_t/128 entries).

## Out of scope

Learned/dynamic predictors (Phase 4), cross-engine ports, OasisKV-style
draft-prefetch (needs MTP drafts to predict rows — revisit **after** 2b-4
shows real acceptance rates; that combination is noted in
`docs/RESEARCH_MAP_DENSE.md` and would compose cleanly with this scan set).

## Implementation status (2026-10-20, shipped)

`engine/scan_decay.h` (knobs, builder, report) + `tools/patch_scan_decay.py`
(12 anchored edits, backup `qwen36.c.prescan`, chain 16 hops, CI replays).

### Amendments to the contract (both forced by measurement)

1. **The salience table is uint64 fixed-point (x 2^14), not f32.** The flash
   prefill seeds under `schedule(dynamic)`, so per-thread partial sums group
   differently run-to-run; float addition is not associative, so an f32
   table -- and with it the scan set and the generated ids -- would vary.
   Integer sums are order-associative: the merged table is bit-reproducible
   under any scheduling (gate 1c proves it end-to-end, prefill included).
2. **Seeding scope: settle row + prefill chunks, no snapshots.** Decode mass
   accumulates only from the settle row (s == 0) -- the one row of a verify
   batch that always stands -- so the q36_spec snapshot/rollback interaction
   (correctness note 1) vanishes entirely: rejected drafts cannot
   contaminate the table. Prefill seeds ALL chunk rows via
   `g_q36_scan_prefill`, set around the chunked prefill driver's call (the
   only `attention()` call with that exact signature). A `pos_base == 0`
   gate would have seeded only chunk 1 of the prompt -- the E7 progress
   work made prefill chunked; the driver flag is the correct signature.

### Gates (synthetic, measured)

| gate | result |
|------|--------|
| 1 mechanical: `Q36_SCAN` unset id-identity vs pre-scan binary (kv8 + tq4, 15,960-token prompt) | **PASS** (bit-identical, the `!scan` path is the 3a/3b loops verbatim) |
| 1 mechanical: two-tier determinism, two runs, full prefill seeding | **PASS** (DUMP_IDS identical) |
| machinery ceiling: `Q36_SCAN_EPS=1.0` (window+sinks only) | **3.2% rows scanned, attention 16.22 -> 1.36 ms/tok (11.9x)** at 16k |
| engagement: `EPS=0.05`, `WIN=512` | **54.1% rows, salience kept 0.9575, attention 16.22 -> 13.92 ms/tok (-14.2%)** |
| 2 salience self-check (repeated-ngram prompt) | **documented NEGATIVE on synthetic** -- see below |
| machinery ceiling at the **68.4k fixture** (`EPS=1.0`) | **2.4% rows, attention 73.66 -> 5.25 ms/tok (14.0x)** |
| engagement at 68.4k (`EPS=0.05`) | **45.6% rows, kept 0.9564, attention 73.66 -> 64.49 ms/tok (1.14x)** -- see the gate-4 verdict |
| 3 quality / 4 speed-on-real-weights | pre-registered Phase 5 arms (unchanged) |

**The gate-2 negative, precisely.** On the random-weight container every
(layer, kv-head) table row shows the SAME top blocks with values in a
perfect arithmetic sequence (28431124992, 27894254080, ... -2^29 per block):
the salience is a pure **causal-coverage gradient** (block b collects mass
from every query after it; earlier blocks collect more). Identical K rows
in distribution carry zero content information, so repeated ngrams cannot
concentrate mass: ngram blocks landed in the EPS=0.5 scan set at 6/14 vs a
~50% null. The mechanism deterministically tracks the mass it sees -- on
random weights there is no content signal to track. Gate 2 becomes
measurable only on real weights (Phase 5) or a container with structured
K rows (a small generator extension: cluster a token subset along a
direction so repeats produce genuinely higher dots -- noted as follow-up).
The knob therefore stays default `full`, exactly as the contract's escape
hatch requires.

**The gate-4 synthetic verdict (68.4k fixture, all arms rc=0).** The
machinery ceiling is real: `EPS=1.0` (window + sinks, 2.4% of rows, a
mostly-CONTIGUOUS set) cuts attention 73.66 -> 5.25 ms/tok (**14.0x**).
But the honest `EPS=0.05` engagement returns only **1.14x** at 45.6% of
rows: the salience-gated set is scattered blocks across a 2.1 GB KV pool,
and random-ish 8 KB chunk reads cost most of what the halved row count
saves -- the full scan is pure sequential streaming and hard to beat at
45% coverage. Gate 4 is therefore a documented negative on synthetic at
`EPS=0.05` (contract: "any fail = documented negative, knob stays full").
The real question is Phase 5's: real weights concentrate mass far harder
than random (the whole premise of the salience gate), which shrinks the
set toward the contiguous-window regime where the 14x ceiling lives --
and if they do not, the lever is dead for this architecture and this file
records why. Full numbers: `results/results_scan_decay_q36.json`.
