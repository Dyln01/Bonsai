# GQA-absorbed decode tiles (Phase 3a) — qwen36.c

**Status: shipped** (2026-10-19). `tools/patch_gqa_absorb.py`, 5 anchored
edits, backup `engine/qwen36.c.pregqa`. No knobs — a pure restructure of the
quantized-KV decode path, active whenever KV8 or KV_TQ is on. Chain link:
`prei4 -patch_mtp_i4-> pregqa -patch_gqa_absorb-> qwen36.c` (verify_engine.sh).

## The bug-shaped finding

The decode branch (S ≤ 8, the flash-off path) parallelized over
`(query-head, row)` pairs and dequantized its K/V tiles **inside** that
loop — so every head of a GQA group re-dequantized the *same* tiles, once
per head. Under f32 KV nobody noticed: there is no dequant there, the dot
reads f32 directly. Under KV8 — and especially under KV_TQ, whose decode
is a Hadamard unrotate plus codec per row — that redundancy was most of the
quantized tiers' decode cost:

| KV tier | dequants per tile | q_per_kv (synth_qwen27 / real 27B) |
|---|---|---|
| f32 | 0 (dot reads f32) | — |
| KV8 | 2× / 5× more than needed | 2 / 5 |
| KV_TQ4 | 2× / 5× more than needed | 2 / 5 |

## The fix

Restructure the loop to `(kv-head, row)`: each K tile and V tile is
dequantized **once** into the per-thread staging and serves every head of
its group (`for g in 0..q_per_kv`). New scratch:

- `m->gqa_sc` — `[attn_sc_thr][q_per_kv][kv_cap]` per-head score rows of
  one group (allocated beside `attn_sc` in the kv-grow path; NULL on
  adapters that never grow — those fall back to the original loop via the
  `m->gqa_sc` guard, so knob-off engines see zero code-path change).
- `kv8_stage` grows by `q_per_kv·kvd` floats per thread: the head-group
  output accumulators (`cxg`) live in that tail, at the per-thread stride
  `(2·TILE_ROWS + q_per_kv)·kvd`.

Lossless by construction: the same dots over the same dequantized values,
the same softmax over the same score row, the same t-ascending V
accumulation — only the dequant count drops.

## Gates (all green)

1. **Identity, A/B binaries**: `DUMP_IDS` full-stream md5 (prompt +
   generated ids) identical between the pre-patch and absorbed binaries —
   f32 (control), KV8, KV_TQ4 at 15,960-token context, and KV8 at the
   68,400-token fixture (the largest in the house; full length — attention
   cost scales linearly ~4.3× vs 16k, confirming O(T) at the true ctx).
2. All 8 battery arms rc=0; `mtp-kv8` combined smoke rc=0 (Phase 2's head
   drafting and the absorbed decode coexist).
3. f32 control unchanged (its path is untouched; +2.8% is inside the
   ±5–7% identical-config noise band).
4. Chain replay: `verify_engine.sh` — `prei4 -> pregqa [patch_mtp_i4.py]`,
   `pregqa -> FINAL [patch_gqa_absorb.py]`.

## Measured (synth_qwen27, 4 threads, attention ms/token, 48 new)

| arm | pre | post | delta |
|---|---|---|---|
| f32 @15,960 | 10.95 | 11.26 | noise |
| **KV8 @15,960** | 19.76 | **12.95** | **−34.5%** |
| **KV_TQ4c1 @15,960** | 96.37 | **51.03** | **−47.0%** |
| **KV8 @68,400** | 79.39 | **57.65** | **−27.4%** |

RAM unchanged per arm (the `gqa_sc` scratch is ~2.2 MB at 68k ctx).
Full numbers: `results/results_gqa_absorb_q36.json`.

The −47% TQ4 / −34.5% KV8 cuts approach the q_per_kv=2 redundancy ceiling
(2×); the remaining cost is real per-row work (dots, and for TQ the codec).
On the real dense 27B (q_per_kv = 5) the ceiling is 5×.

## What this does NOT fix

- **TQ's residual ~4× gap vs KV8** (51.0 vs 12.95 ms/tok): the per-row
  FWHT unrotate is genuine work, not redundancy. Closing it needs a true
  absorbed-score / MLA-style formulation (Phase 3b) or a cheaper codec —
  this patch only removed the redundant invocations.
- **O(T) scan decay** at long ctx (57.65 ms/tok at 68k): the per-token
  score walk is unchanged. Phase 3b territory (absorbed score over a
  compressed index), not a loop restructure.

## Lessons

1. **Per-thread scratch stride must match the allocation stride.** The
   first build enlarged `kv8_stage` per-thread but kept indexing threads
   at the old `2·TILE·kvd` stride — each thread's accumulator tail landed
   in the *next thread's* K staging. Symptom: KV8/TQ4 id streams diverged
   (f32 control passed!) while every arm still exited rc=0 — cross-thread
   corruption is silent. The A/B identity gate caught it in one run.
2. **A/B identity against a pre-patch binary is the cheapest real gate**
   for "pure restructure" claims: build both binaries, same env,
   `DUMP_IDS`, compare. It separates "bit-identical, cheaper" from
   "plausible-looking" instantly.
3. **A +2.8% control wobble on an untouched path is the noise band** —
   report it, don't hide it; it calibrates every other delta in the table.
4. Windows checkout CRLF strikes again: the repo's `.sh`/`.c` working
   files carry `\r\n` (autocrlf) while the blobs are LF — byte-anchored
   patchers and verify scripts must run against LF content (CI does;
   locally, normalize before anchoring).
