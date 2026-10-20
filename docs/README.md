# docs/ — index & status

Start here: [INSTALL.md](INSTALL.md) (setup) · [COMMANDS.md](COMMANDS.md) (every knob) ·
[ROADMAP.md](ROADMAP.md) (the phase board). Status vocabulary: **shipped** = in the engine +
measured; **measured** = harness result, no engine change; **negative** = measured, did not
ship (kept so we don't retry blindly); **design** = contract pinned, implementation open.

| Doc | Date | Status | One line |
|---|---|---|---|
| [INSTALL.md](INSTALL.md) | 2026-10-17 | shipped | Linux-native + Windows/WSL2 install, models, verification, troubleshooting |
| [COMMANDS.md](COMMANDS.md) | 2026-10-17 | shipped | launcher subcommands, ~30 engine knobs, wire protocol, generators, patch order, parity gates |
| [ROADMAP.md](ROADMAP.md) | 2026-10-19 | living | the phase board + open engineering items |
| [VALIDATION_PROTOCOL_QWEN36.md](VALIDATION_PROTOCOL_QWEN36.md) | 2026-10-20 | sign-off pending | Phase 5 contract: checkpoints, eval data, arms with pre-registered acceptance bands, mini-PC mechanics, session procedure |
| [REPORT.md](REPORT.md) | 2026-10-11 | measured | the original harness deep-dive (MoE synth containers, I/O throttle, pin/LRU); see its errata header for path drift |
| [RESEARCH_MAP_DENSE.md](RESEARCH_MAP_DENSE.md) | 2026-10 | measured | literature map: what we took, what we deliberately didn't (windowing, dynamic predictor, OasisKV) |
| [TRUNKMAP_DENSE27B.md](TRUNKMAP_DENSE27B.md) | 2026-10 | shipped | mmap'd trunk + in-register dequant in the GEMV |
| [HOTNEURON_DENSE27B.md](HOTNEURON_DENSE27B.md) | 2026-10 | shipped | hot-neuron paging, HOT_FRAC exactness dial |
| [HOTNEURON_V4_CALIBRATION.md](HOTNEURON_V4_CALIBRATION.md) | 2026-10 | shipped | self-calibration pipeline (`hotcal_pipeline.py`) |
| [GRIFFIN_V5.md](GRIFFIN_V5.md) | 2026-10 | shipped | per-prompt neuron selection, training/calibration-free |
| [MINIPC_QWEN30B.md](MINIPC_QWEN30B.md) | 2026-10 | measured | MoE 30B-class runs on the mini-PC envelope |
| [MINIPC_V2_INTEGRATED.md](MINIPC_V2_INTEGRATED.md) | 2026-10 | shipped | v2 integration of the stack into one launcher |
| [Q36_MINIPC_TOOL.md](Q36_MINIPC_TOOL.md) | 2026-10 | shipped | the launcher itself (`tools/q36-minipc.sh`) |
| [KV8_QWEN36.md](KV8_QWEN36.md) | 2026-10 | shipped | fp8-e4m3 KV tier, parity-proven 256/256 codes; decode cost labeled |
| [FLASH_QWEN36.md](FLASH_QWEN36.md) | 2026-10 | shipped | flash-tile prefill, 2.4x TTFT cut; S=1 limit documented |
| [LOWRAM_QWEN30B_FINAL.md](LOWRAM_QWEN36_FINAL.md) | 2026-10 | measured | final low-RAM matrix + KV8 decode version progression |
| [KVTQ_QWEN36.md](KVTQ_QWEN36.md) | 2026-10-16 | shipped | third KV tier (PolarQuant / rotated-int4), 8-arm battery, RAM-vs-decode honesty table |
| [PREFIX_REUSE_QWEN36.md](PREFIX_REUSE_QWEN36.md) | 2026-10-16 | shipped | transcript KV-prefix reuse across agent turns (93%, 10-12x turn 2+), agent contract |
| [SOFTMAX_QWEN36.md](SOFTMAX_QWEN36.md) | 2026-10-17 | shipped | Q36_FAST_SOFTMAX AVX2 fast exp; includes two negative results (flash-site, silu) |
| [GQA_ABSORB_QWEN36.md](GQA_ABSORB_QWEN36.md) | 2026-10-19 | shipped | GQA-absorbed decode tiles: dequant once per group, KV8 −34.5% / TQ4 −47% attn ms/tok, bit-identical outputs |
| [BENCH_DENSE27B.md](BENCH_DENSE27B.md) | 2026-10-17 | measured | dense-27B benchmark walkthrough (projections labeled) |
| [SCAN_DECAY_QWEN36.md](SCAN_DECAY_QWEN36.md) | 2026-10-20 | design | Phase 3b-scan: two-tier deterministic decode scan (window + sinks + salience-gated blocks); gates pre-registered |
| [ABSORB_ROT_QWEN36.md](ABSORB_ROT_QWEN36.md) | 2026-10-19 | shipped (absorb branch) | Phase 3b-rot: absorbed-rotation TQ decode (per-row FWHT eliminated); 3b-scan/3b-prefill follow-ups |
| [GQA_ABSORB_QWEN36.md](GQA_ABSORB_QWEN36.md) | 2026-10-19 | shipped | Phase 3a: GQA-absorbed decode tiles (dequant once per group); measured KV8 −34.5% / TQ4 −47% attn |
| [MTP_DESIGN_QWEN36.md](MTP_DESIGN_QWEN36.md) | 2026-10-19 | shipped (test) | MTP head: loader + dense/MoE drafting/verify + oracle proofs + resident int4, all CI-gated; battery on real hardware open (2b-4) |
| [GRIFFIN_V5.md](GRIFFIN_V5.md) / [HOTNEURON_V4_CALIBRATION.md](HOTNEURON_V4_CALIBRATION.md) | — | — | both list absorbed-score attention as the next architectural lever (now Phase 3) |

Negative results, kept on purpose: flash-site online-softmax fast-exp (SOFTMAX §), silu
`expf` fast path (0.96x, SOFTMAX §), KV8 branchy bit-math decode (333 s regression,
LOWRAM §), io_uring under the throttle shim (REPORT §limitations).
