# Field findings — dense 27B on the i5-6400T mini-PC (Nov 16–18, 2026)

What the `q36-minipc bench` logs (written to the Samba share by `tools/q36_benchlog.py`) established about
why decode is hundreds of seconds per token, what has been fixed, and what is still open. Numbers are from
the logs; anything inferred is marked **hypothesis**.

## The machine
- Intel i5-6400T, 4 cores / 4 threads, 2.2 GHz base / 2.8 GHz turbo, AVX2 + FMA, no AVX-512; governor `powersave`.
- 7.62 GiB RAM in **one** DDR4-2133 DIMM (single channel, ~17 GB/s peak).
- Model on a spinning **ST3160815AS** SATA HDD (`rotational=1`, `read_ahead_kb=128`); random reads ≈ 150 IOPS,
  which at ~95 KB per fault is the 13–23 MB/s seen in every thrashing run.
- Ubuntu 20.04, kernel 5.15. Swap: 5.8 GiB, later 2 GiB (both on the same HDD).

## Runs (decode s/token; "read" = disk bytes for the whole run)
| run | engine / knobs | decode | read |
|---|---|---|---|
| default, 16 tokens (hot_check_1–5) | old engine, window 512 MB, GRIFFIN on | 351–366 | 394 GB |
| default, 11 tokens (hot_check_6/7) | same (knob changes never reached the engine) | 325–327 | 338 GB |
| default + AVX2 kernel (Nov 16) | `[mapi4]` active | 328 | 338 GB |
| default + `Q36_NO_VISION` (Nov 17/18, ×2) | vision tower skipped | 317, 320 | ~334 GB |
| mem fix, GRIFFIN on | `EMBED_MAP=1 WIN_MB=0` | 415 | 51 GB |
| mem fix, GRIFFIN off | + `GRIFFIN=0` | 377 | 50 GB |
| frac 0.12, GRIFFIN off | old engine | 494 | 45 GB |
| frac 0.17 / 0.25, "E28" label | **old engine** (no `[down-compact]` line) | 505 / 398 | 43 / 51 GB |
| fit probe frac 0.06 — cold cache | old engine | 486 | 40 GB |
| fit probe frac 0.06 — 21:53 | old engine, same knobs | **52** | 10 GB |
| fit probe frac 0.06 — 14:09 next day | old engine, cache full at start | 531 | 45 GB |

The new AVX2 `matmul_map_i4` kernel changed nothing in these runs: the box is I/O-bound (iowait median
~80 %, 10–25 % of one core busy), so compute speed cannot show until the weights stay in RAM.

## Where the memory goes (the cost model was wrong here)
Default launch: ~**4.9 GB of private (anonymous) RAM after load** = `model-globals` 3.18 GB (mostly the f16
embedding table) + `model-vision` 0.92 GB + 0.5 GB window cache. The floor model assumed ~0.7 GB.
With `Q36_EMBED_MAP=1 Q36_NO_VISION=1 Q36_WIN_MB=0`: anonymous RAM **0.27–0.37 GB**, RSS after load 0.21 GB,
load time 6 s (was 106 s). Kernel + other overhead ≈ 0.67 GB. The page cache then reaches ~7.0 GB
(5.0 GB when 2.1 GB was private).

## The working-set model (validated four times)
`down_proj` is stored `[hidden, inter]`, so the hot neurons are the first `Ik` **columns of every row**: a
prefix read touches most 4 KB pages of the matrix (71 % at frac 0.25; ≥ ~1.3 GB over 64 layers however small
`frac` is) instead of the `0.71 GB` the hot bytes occupy. Per-pass working set ≈ 4.24 GB (DeltaNet, attention,
lm_head) + gate/up prefix + touched `down` pages.

| frac | model | measured bytes read per forward pass |
|---|---|---|
| 0.25 | 7.68 GB | 7.59, 7.44 |
| 0.12 | 6.61 GB | 6.8 |
| 0.06 | 6.10 GB | 5.9 |

(per pass = (total read − warm-up) / 6 passes.) The model predicts the *bytes read*, not whether the page
cache keeps them.

## What is NOT understood
- **Why a working set smaller than the cache still thrashes.** At frac 0.06 the model says 6.1 GB against a
  ~7.0 GB cache, yet two of three identical runs read ~5.9 GB per pass. One run (21:53) had real 52 s forward
  passes reading ~0.66 GB each. It was **not reproduced** by starting with a full cache (14:09: 531 s/token).
  Hypotheses (unproven): readahead pollution from the strided `down` reads; Linux active/inactive list
  behaviour for a cyclic scan longer than the inactive list.
- The generated text is empty in every log (stdout is not captured); quality at the tested fracs is unknown.

## Fixes shipped (all CI-verified; none yet confirmed on the box)
- E26 `mapi4_avx2.h`: AVX2 staging for `matmul_map_i4` (bit-exact), one-row calls skip the thread team,
  opt-in integer dot (`Q36_MAPI4_IDOT=1`).
- E27 `Q36_NO_VISION=1`: skip the 0.9 GB vision tower (bench turns it on).
- E28 `Q36_DOWN_COMPACT=1`: copy the hot `down` columns once into a compact buffer (bit-exact; verified
  over 480 single/batched calls by `scripts/check_down_compact.py`). Predicted to cut the `down` working set
  from ~2.0 GB of page cache to ~0.71 GB at frac 0.25. **No log has run it yet** — every "E28" log so far shows
  the old launcher/engine; check the header's md5s and for a `[down-compact]` line.

## Open work, in order
1. Install the current engine/launcher on the box (`tools/make_bundle.sh` builds a git-free installer) and
   run `Q36_DOWN_COMPACT=1 Q36_GRIFFIN=0 Q36_EMBED_MAP=1 Q36_WIN_MB=0 Q36_HOT_FRAC=0.17`.
2. If residency is still unreliable: `mlock` the hot set (opt-in) instead of relying on kernel LRU heuristics.
3. The decode-time `down` loop (`q36_down_prefix`, mapped int4) is scalar and serial; vectorizing it changes
   the summation order, so it needs a numerics decision.
4. Teach `q36-minipc floor` the corrected accounting (private RAM, ~0.67 GB overhead, touched `down` pages).
5. Expected target once resident: roughly 0.5–1.0 s/token on this single-channel box (estimate, unmeasured).
