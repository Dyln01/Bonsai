# results/ — measured JSON manifest

Every number quoted in `docs/` traces to one of these files (honesty chain:
**doc table → driver script → JSON here → commit**). Synthetic arms use
same-geometry containers; the field-run section below carries the real-weights
llm-030 sessions. Parsed by CI on every push.

| JSON | Driver | Report | Notes |
|---|---|---|---|
| `results_std.json` | `scripts/run_workloads.py` (cold/steady/long profiles) | `docs/REPORT.md` | std synth container baseline |
| `results_kv_serve.json` | `scripts/exp_kv_serve.py` | `docs/REPORT.md` | KV pool at CTX=131072, SERVE mode |
| `results_kv_trunk.json` | `scripts/exp_kv_and_trunk.py` | `docs/REPORT.md` | dense-trunk offload A/B |
| `results_pin.json` | `scripts/exp_pin_recovery.py` | `docs/REPORT.md` | learned-pin residency recovery |
| `results_qwen_kv8.json` | `scripts/exp_qwen27_kv8.py`, `exp_qwen27_kv8b.py` | `docs/KV8_QWEN36.md`, `docs/LOWRAM_QWEN36_FINAL.md` | KV8 arms incl. decode version progression |
| `results_qwen_final.json` | `scripts/exp_qwen27_final.py` | `docs/LOWRAM_QWEN36_FINAL.md` | final low-RAM matrix |
| `bench_q36.json` | `q36-minipc bench` (launcher battery) | `docs/BENCH_DENSE27B.md` | phase-split honest battery |
| `results_kvtq_q36.json` | `scripts/exp_kvtq_q36.py` | `docs/KVTQ_QWEN36.md` | the 8-arm KV_TQ matrix (RSS ladder + decode prices) |
| `results_prefix_q36.json` | `scripts/exp_prefix_q36.py` | `docs/PREFIX_REUSE_QWEN36.md` | reuse % through mux slot swaps |
| `results_softmax_q36.json` | `scripts/exp_softmax_q36.py` | `docs/SOFTMAX_QWEN36.md` | 5/5 pairs A/B + microbench |

Convention: decode tok/s only from the engine's `[timers] TOTAL ms/token` line;
`Speed:` lines include prefill and are never quoted as decode. New batteries
append here + cite the JSON in the doc that uses them.

## Field runs (real weights — llm-030, 8 GB RAM / HDD mini-PC)

| File | Runner | Verdict |
|---|---|---|
| `battery_llm-030_20261022-192421.json` | run_battery.sh pre-`8149f4a` | INVALID — runner hardcoded MAXT 8192 vs a 15,960-token prompt; all arms rc=1. Kept as a negative result. |
| `battery_llm-030_20261022-193000.json` | run_battery.sh pre-`8149f4a` | INVALID — frac 1.0 + f32 KV at 16k exceeds the 8 GB envelope by arithmetic; all arms rc=137. |
| `battery_llm-030_20261023-222216.json` | run_battery.sh pre-E11 | INVALID — runner launched the STOCK dialect (no `TRUNK_MAP=1`; `Q36_HOT_FRAC` alias inert): seven arms rc=137 at an identical ~5.10 GB fs-read watermark = arm 1 of `docs/BENCH_DENSE27B.md` re-proved nine times (ROADMAP E11). E/F skipped cleanly. |
| `battery_llm-030_20261024-114815.json` | run_battery.sh post-E11 | INVALID for numbers — smoke arm A died rc=124 at the hardcoded 3600 s wall mid-prefill (ROADMAP E12). VALID as the E11 field proof: `TRUNK_MAP=1`+stream, peak RSS 7.0 GB < 7.62 GB total, no signal-9 kill. |
| `battery_llm-030_20261024-133054.json` | run_battery.sh post-E11 | INVALID for numbers — 7 compute arms each ran the full 3600 s and were killed (fs_read 37-42 GB ≈ one cold pass at frac 0.25 → 10.3-11.6 MB/s effective); E/F skipped cleanly. The measured HDD budget is now in the runner (`BATTERY_TIMEOUT`, budget echo, `_logs/` persistence). |
| `battery_llm-030_20261025-101318.json` | run_battery.sh post-E12 | **VALID — the first completed field arm in project history** (E12 probe, arm A, n_new=2, frac 0.25, maxt 8192, stream 1, timeout unlimited): rc=0, wall 1:40:06, zero swaps, peak RSS 6.94/7.62 GB, TTFT 3676.22 s (1524-token prompt), decode 2,225,442 ms/token (`[timers] TOTAL`, one decode step; MLP trunk = 71.7%, attention core 17.2 s), fs_read 71.72 GB. Generated `чисwik` = documented unsorted-container pruning (quality deferred to hot_sort + phase-5 PPL), not a fault. E/F clean field skips. Cost model: T(arm) ≈ 105 s + 3676 s + (n_new−1)×2225 s (ROADMAP E13). |
| `realweights_20261022-2011/manifest.txt` | run_phase5.sh preflight | VALID — 66 shard sha256s of `/root/Models/q27_i4` (conversion provenance). Only the preflight manifest was committed; any pre-E11 arm `.err` on the box is tainted (decoy `Q36_*` aliases → every arm ran frac 1.0/griffin 0) — re-run with `PHASE5_FORCE=1`. |

Kept per house rules: negative results documented, not buried. As of
2026-10-25 the first VALID field numbers exist (the E12 probe row above —
quoted in ROADMAP E13 and the `docs/BENCH_DENSE27B.md` arm-1 addendum, both
citing that JSON). The re-budgeted 6-arm matrix (B/C/D + the W read-matrix,
~10 h overnight at n_new=2) and the phase-5 summary append here when they
land.
