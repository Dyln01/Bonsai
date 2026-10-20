# Decode scan decay (Phase 3b-scan) — design contract

**Date:** 2026-10-20 · status: **design pinned, implementation next**
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
