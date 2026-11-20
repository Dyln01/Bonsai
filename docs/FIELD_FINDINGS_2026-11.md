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
  over 480 single/batched calls by `scripts/check_down_compact.py`). **Confirmed on the box Nov 19-20 — see
  "E28 on hardware" below.**

## E28 on hardware (Nov 19–20): the resident regime is reached
Ten runs with E28 active (launcher `9c4935…`, `[down-compact]` line present; `GRIFFIN=0 EMBED_MAP=1 WIN_MB=0`,
`kv8`). Compaction holds all 64 layers: 462 MB at frac 0.17, 680 MB at 0.25, 816 MB at 0.30.

| frac | tokens | total wall | TTFT | disk read | I/O | tok/s overall | tokens/forward | T(1) ms | down-compact held |
|---|---|---|---|---|---|---|---|---|---|
| 0.17 | 6 | 294 s | 271.64 s | 8.0 GB | ok | 0.02 | 1.00 | 2628.6 | 462 MB |
| 0.17 | 64 | 303 s | 177.60 s | 4.9 GB | ok | 0.21 | 1.66 | 2725.3 | 462 MB |
| 0.17 | 128 | 483 s | 205.32 s | 5.8 GB | ok | 0.27 | 1.67 | 3277.0 | 462 MB |
| 0.20 | 64 | 468 s | 189.44 s | 6.9 GB | ok | 0.14 | 1.37 | 5380.3 | 544 MB |
| 0.22 | 64 | 470 s | 158.49 s | 7.8 GB | ok | 0.14 | 1.37 | 6202.5 | 598 MB |
| 0.25 | 64 | 499 s | 175.50 s | 8.3 GB | ok | 0.13 | 1.66 | 16079.2 | 680 MB |
| 0.25 | 128 | 609 s | 158.54 s | 7.6 GB | ok | 0.21 | 1.81 | 15194.1 | 680 MB |
| 0.30 | 48 | 435 s | 162.54 s | 7.0 GB | ok | 0.11 | 1.47 | 11080.8 | 816 MB |
| 0.17 | 256 | 710 s | 134.45 s | 5.2 GB | ok | 0.37 | 1.72 | 4357.7 | 462 MB |
| 0.25 | 256 | 718 s | 102.87 s | 3.4 GB | ok | 0.36 | 1.90 | 5122.1 | 680 MB |

- **Thrash is gone.** Disk read 3.4–8.3 GB per run (was 40–50 GB), I/O verdict "ok" in every run. After a
  one-time warm-up of ~200–240 s (disk ~30 MB/s while ~5 GB of trunk is read once) the long runs sit at
  **0.0 MB/s disk, 0 major faults, 399 % CPU, 2.5 GHz steady** — fully resident and compute-bound.
  Decode went from ~400–500 s/token to ~2.3 s/token.
- **End-to-end speed** (256 tokens): frac 0.17 → 710 s total, TTFT 134 s, so **2.26 s/token** after the first
  token (0.37 tok/s overall incl. prefill); frac 0.25 → 718 s, TTFT 103 s, **2.41 s/token**.
  Raising frac 0.17 → 0.25 (+50 % MLP weights) costs only ~7 % per token: **the MLP is not what limits speed.**
- `T(1)` in the table is measured early in decode while the cache is still warming at the higher fracs
  (16 s at 0.25 vs 2.6 s at 0.17), so it is not a steady-state number. Short runs (6/64 tokens) are dominated
  by warm-up: use 256 tokens or more.
- Lookup speculation gives 1.4–1.9 tokens per forward.
- **Compute is slower than the kernels explain.** The long runs are 99.5 % user time (2,183 s user, 10 s system),
  i.e. ~1.2 ns of CPU time per weight, while the staged matmul benchmarks at 0.135–0.19 ns/weight with
  colibri's real `dot_f32_lanes` (`-O3`, skylake/v3/generic). So most of each ~2–4 s forward is in code outside
  `matmul_map_i4`. The engine's phase timers (first 19 tokens, still warming) put DeltaNet (2.6 s/token) and the
  MLP bucket (2.4 s/token) first; the scalar serial `q36_down_prefix` loop is one known suspect, DeltaNet
  another. **Unmeasured — run `tools/q36_profile.sh` on a 256-token run.**

## Open work, in order
1. **Profile the steady state** (`bash tools/q36_profile.sh` in a second terminal during a 256-token
   `q36-minipc bench`): it ranks functions by time and names what the ~2 s/token actually is.
2. Fix whatever dominates. Known candidates: the scalar serial `q36_down_prefix` (needs a numerics decision —
   vectorizing changes the summation order), and the DeltaNet path.
3. `mlock` the hot set (opt-in) if residency ever proves unreliable on other boxes; with E28 it was stable here.
4. Teach `q36-minipc floor` the corrected accounting (private RAM, ~0.67 GB overhead, touched `down` pages).
5. Target: ~1 s/token. Now 2.3 s/token at frac 0.17–0.25 on this single-channel box.
