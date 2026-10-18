# results/ — measured JSON manifest

Every number quoted in `docs/` traces to one of these files (honesty chain:
**doc table → driver script → JSON here → commit**). All runs are synthetic
same-geometry containers; no real weights. Parsed by CI on every push.

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
