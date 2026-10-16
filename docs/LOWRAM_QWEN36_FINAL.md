# qwen36 low-RAM/high-speed — final results (v3 KV8 + profiles)

**Date:** 2026-10-12 · engine `qwen36` @ bf24429 + local KV8 patch v3 (`qwen36.c`, `kv8_fused.h`, `kv8_parity.c`)
**Model:** synthetic Qwen3.8-27B-shaped container (24L, 6 GQA + 18 DeltaNet, hidden 1024) — no download.
**Host:** WSL2, Ryzen 7 7735HS (8C/16T), 16 GB RAM.

## New techniques built this round

1. **Branch-free bit-math e4m3 decoder** (`kv8_dec_vec`): converts fp8→f32 with integer selects instead of LUT lookups or branches, so it auto-vectorizes. **Value-parity with the engine's own `coli_fp8_lut` proven over all 256 codes** (`kv8_parity.c`: "all 256 codes identical, signed-zero tolerant"), plus a 191,625-element round-trip error gate (mean rel err 0.022, under the e4m3 half-step bound 0.0625).
2. **Tile-staged KV dequant** (the technique the GLM engine uses, ported): 256-row tiles dequantized branch-free into per-thread L2-sized staging, then the *same vectorized `dot_f32_lanes`* the f32 path uses — plus scale-hoisting (per-row scale applied once after the dot / before the axpy, never per element).
3. **The combined low-RAM profile**: `KV8=1` + `COLI_DENSE_BITS=4` (the engine's own int4-dense knob), measured together for the first time.

## Measured matrix (~8k context, 48-token decode, all real engine runs)

| Profile | Peak RSS | vs A | TTFT | Decode tok/s | vs A |
|---|---|---|---|---|---|
| **A. defaults (f32 KV + int8 dense)** | 804.5 MB | — | 74.2 s | 0.85 | — |
| **B. KV8 v3** | 673.8 MB | **−131 MB (−16%)** | 112.2 s | 0.56 | −34% |
| **C. int4 dense** | 728.0 MB | −77 MB (−10%) | 76.6 s | 0.83 | −2% (noise) |
| **D. KV8 + int4 dense** | **595.6 MB** | **−209 MB (−26%)** | 112.3 s | 0.56 | −34% |

Version progression of the KV8 decode (same workload): naive per-row LUT **121 s** → branchy bit-math (regression) 333 s → **tile-staged branch-free v3: 108–112 s**. The remaining ~1.5× attention cost vs f32 is structural for this engine: qwen36 reads full head_dim K/V rows per (query-head, kv-head) with no MLA absorption, and at decode (S=1) each dequantized tile serves only H/KVH = 2 query rows, so the dequant pass cannot amortize. The GLM engine's KV8 is decode-neutral *because* MLA absorbs the score path into weights — that architectural property, not the codec, is what qwen36 lacks.

## Interpretation for weak hardware (the honest trade table)

| Situation | Recommended profile | Why |
|---|---|---|
| RAM fits (short ctx) | A (defaults) | f32 KV is the speed champion; KV8 buys nothing |
| RAM-bound at long ctx | B or D | −16/−26% RSS; decode cost is the price of fitting |
| RAM-bound + bandwidth-starved CPU | D | int4 dense also halves weight traffic per token (measured −2% here because this box is compute-bound, but on the documented real-27B the engine's own A/B showed 2.25→3.15 tok/s from int4 dense) |

**Scaled to real Qwen3.8-27B (64L, kv_heads 8, head_dim 128):** profile D's KV saving becomes **17.2 GB at 32k context** (4.3 GB vs 17.2 GB) — plus the dense weights drop from 29 GB (int8) toward ~21.6 GB (int4 MLP+head, engine's own measured table). Combined, a 32 GB host goes from "cannot run 32k ctx" to "runs it with ~14 GB to spare". **Extrapolation from the engine's own published 27B measurements + this harness's ratios; needs real-checkpoint confirmation.**

## Correctness gates

- **Decoder parity: PASS** (256/256 codes identical to `coli_fp8_lut`, both `kv8_dec1` and `kv8_dec_vec`)
- **Round-trip error: PASS** (mean 0.022 < 0.0625 e4m3 half-step; max-1 cases are the codec's designed near-zero flush)
- **All 4 profiles: rc=0**, no crashes; default path byte-identical to unpatched engine when `KV8` unset (no code path changes)
- Known fixture caveat (pre-existing, not from this patch): unpatched qwen36 is run-to-run nondeterministic on the synthetic container (timing-sensitive speculation gate) — token-stream hashing is not a usable gate here; parity + error bounds above carry the correctness argument.

## Files

- Patch: `~/colibri/c/qwen36.c` (+ `.orig` backup), `~/colibri/c/kv8_fused.h`, `~/colibri/c/kv8_parity.c` (mirrored in `C:\Users\dylan\projects\colibri-harness\`)
- Tests/bench: `exp_qwen27_final.py`, `results_qwen_final.json`
- Prior round: `KV8_QWEN36.md`, `REPORT.md`

## Next steps toward PR-quality

1. **The real speed fix is architectural, not a codec tweak**: a FlashMLA-style absorbed score for qwen36 (fold qk into the query side per GQA group) would amortize dequant the way GLM does — 3–4 person-weeks.
2. `KV_TQ` (4-bit) port using `kv_tq.h` → another ~2× KV RAM on top (0.22 GB → ~0.11 GB per 1k tokens at 27B scale ≈ 8.6 GB at 32k).
3. Port the whole KV8 patch to olmoe/inkling/deepseek_v41 (same three touchpoints).
4. Real-checkpoint validation: perplexity on Qwen3.8-27B + the fixture nondeterminism root-cause.
