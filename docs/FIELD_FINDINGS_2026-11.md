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

## Quality measured (Nov 23, `q36-minipc quality … 0.17 0.25 1.0`, 128 scored tokens, f32 KV, GRIFFIN off)
| frac | TF-NLL | ppl | vs frac 1.0 | wall |
|---|---|---|---|---|
| 1.0 (full compute) | 1.535 | **4.64** | reference | 2066 s (streams ~12.8 GB per weight pass, 3 passes) |
| 0.25 | 7.693 | 2192.84 | x473 — GARBAGE | 706 s |
| 0.17 | 11.323 | 82711.42 | x17,800 — GARBAGE | 648 s |

- **Full compute is healthy** (4.64 on the probe text; the repo's wikitext reference is 6.74 — different texts, compare
  rows to each other). So the engine, the container, the sort/pack and E26–E29 are correct at scale (the AVX2
  staging is active at frac 1.0, and the result is sane).
- **Static hot-prefix pruning to <= 25 % destroys the model even on the SORTED container.** 0.25 is x473 the
  reference here versus x266 for the repo's UNSORTED measurement (ppl 1791 vs 6.74) — different texts, so not a
  proof, but there is no visible benefit from sorting at this fraction.
- **Consequence:** every speed number at frac <= 0.30 (1.22 s/token, resident, 399 % CPU) describes a model that
  emits garbage. The bench text being degenerate in every log was this, not a bug.
- **The bind on this box:** residency needs frac <= ~0.3 (4.24 GB non-MLP + MLP hot set in ~7 GB of page cache);
  usable quality needs far more (unsorted 0.5 was x4.3 ppl; sorted 0.5 untested). Full compute is 12.8 GB of int4
  weights: it does not fit in 7.6 GiB.

**Launcher default changed (Nov 23):** `chat`, `serve` and `bench` used to default to frac 0.25 (levels: 0.25 / 0.25 /
0.5), i.e. garbage. The default is now **1.0 (exact full model)**; any `Q36_HOT_FRAC < 1` prints a warning with the
numbers above. Full compute on this box is slow (the whole 12.8 GB trunk streams from the HDD each pass) but correct.

Estimates (not measured) for the options:
- Measured resident compute rate ~6.5 G weights/s (T(1) 1.74 s for ~11.4 G weights at frac 0.17). Full model =
  25.6 G weights -> ~3.9 s per forward -> ~2.3 s/token at 1.69 tokens/forward with today's kernels.
- Memory-bandwidth floor for 12.8 GB per forward: ~1.1 s single-channel (~12 GB/s real), ~0.5 s dual-channel.
  The gap between 3.9 s and 0.5-1.1 s is the compute/threading inefficiency the profile points at (70 % of thread
  samples outside named kernels).
- Full model resident needs ~12.8 GB weights + ~1 GB private + ~0.7 GB kernel = ~14.5 GB: it fits in 16 GB, not 8.

## E29 baseline (Nov 21, f=0.17, 256 tokens) — and a quality problem that outranks every speed number
Launcher `c9aaf9…`, engine built just before the run (`[down-compact]` + `[mapi4]` lines present).
- Total 579.3 s (0.44 tok/s). TTFT 268.7 s (cold page cache: 8.6 GB read at 14.6 MB/s, 62 k major faults).
  Decode after the first token: 310.6 s for 255 tokens = **1.22 s/token** (151 forwards, 2.06 s each, 1.69
  tokens/forward, lookup acceptance 71 %). `T(1)` = 1.74 s.
- Phase timers over the first 19 decoded tokens — the cache was already warm, so these are the first clean
  steady-state numbers: DeltaNet 1.05, attention 0.21, MLP (`moe_total`) 0.63, lm_head 0.12 = 2.00 s/token.
- Versus the last pre-E29 long run (700 s, 2.26 s/token) E29 gains 1.5–1.9x: that run's decode window still
  contained ~100 s of cache warm-up, so the exact factor is uncertain. The kernel-level estimate (7–9 %) was
  far too low — the profile's 25 % inclusive share for `q36_down_prefix_batch` was the better guide.

**Quality.** The "generated text" in every bench log is degenerate (`:: ... . .`, `1 ........ / /`,
`Z: / / / --2222…`) — including the Nov 16 default run (frac 0.25, GRIFFIN on) that exercised none of
E27–E29. The repo's own record (README, `docs/REALWEIGHTS_QWEN36.md`): full compute ppl **6.74**; UNSORTED
frac 0.5 / 0.25 → ppl 29.24 / 1791.28 ("documented negatives"); the SORTED-container arm E was never
measured. The container in use is sorted and still produces garbage at fracs 0.17–0.30, so either
(a) pruning to 17–30 % is too aggressive for this model even with sorted neurons, or (b) the sort/pack is
wrong and even frac 1.0 would be broken. These need different fixes. `q36-minipc quality <model> 0.17 0.25 1.0`
measures it (teacher-forced perplexity through the engine's PPL=1 path, f32 KV, GRIFFIN off) and flags a dead
trunk (ppl == vocab size). **Until it is run, 1.22 s/token is speed at an unmeasured — probably unusable — quality.**
Consequence if (a): residency needs frac <= ~0.3 (working set 4.24 GB + MLP), but acceptable quality may need
>= 0.5, which does not fit in ~7 GB of page cache; other levers (int3 for the non-dial 4.24 GB, fewer layers)
would be needed.

## First steady-state profile (Nov 21, f=0.17, 256 tokens, gdb sampling, 436 thread samples)
`tools/q36_profile.sh` (profiled run: 818 s vs 700 s unprofiled — sampling pauses the process):
- Innermost: **69.5 % in unresolved frames ("??")**, 15.4 % `matmul_map_i4`, 14.2 % `q36_down_prefix_batch`,
  0.9 % `deltanet`. Inclusive: `q36_down_prefix_batch` 25 %, `matmul_map_i4` 31 %.
- Only ~30 % of thread time is in named compute kernels. The unresolved 70 % is most plausibly libgomp's
  internal spin-wait (hypothesis — the first version of the profiler dropped the library name; it now labels
  unresolved frames `?? [library]`). If so, the four threads spend most of their time waiting for each other
  (CPU shows 399 % because spinning counts as busy), not computing.
- `q36_down_prefix_batch` was the biggest named hotspot and still used the original scalar staging loop —
  although down_proj is only ~8 % of the weights read per token. It is the **only** down routine the decode path
  calls (one call site). E29 fixes it (bit-exact).
- At the real shape (5120 rows, Ik 2959, S=1) E29 measures 20.7 → 4.4 ms per call (4.7x; S=4: 2.4x), i.e. about
  0.25-0.3 s off a forward of ~3-4 s (~7-9 %). The profile's 25 % inclusive share is larger than the kernel
  alone explains, which again points at threads waiting around this region (stragglers at barriers).

## Open work, in order
0. **Decide the path** (quality is now measured): the pruning dial cannot reach usable quality at a resident
   fraction. Options: 16 GB RAM (second DIMM; also dual-channel) for full-compute 27B; a smaller model that
   fits int4 in ~7 GB; stronger compression than int4 (research); or retraining-free tricks (per-layer schedule
   `q36-minipc fracs`, GRIFFIN) which are unlikely to close a x473 gap.
1. If full compute becomes resident: kernel/threading work has large headroom (3.9 s/forward estimated vs a
   0.5-1.1 s bandwidth floor): re-profile with the library-labelling profiler, `Q36_THREADS=3`, DeltaNet path.
2. `mlock` the hot set (opt-in) if residency proves unreliable; teach `floor` the corrected accounting.
