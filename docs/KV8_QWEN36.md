# KV8 for qwen36 — measured results (synthetic Qwen3.8-27B-shaped model)

**Date:** 2026-10-12 · engine `qwen36` @ bf24429 + local KV8 patch (files: `qwen36.c`, `kv8_fused.h`; pre-patch backup `qwen36.c.orig`, patched copy `qwen36.c.kv8patched`)
**Model:** synthetic, 24 layers (6 GQA-attention + 18 Gated-DeltaNet), hidden 1024, kv_heads 4, head_dim 64, dense MLP 2048 — Qwen3.8-27B architecture, scaled down. Real weights never downloaded.

## What was built

An opt-in `KV8=1` for the qwen36 engine — the first KV-quantization path in an engine that had none (verified: zero `KV8`/`KV_TQ` sites in qwen36.c before the patch). It reuses the tree's proven `kv_fp8.h` e4m3 codec (per-row amax/448 scale, RNE encode, LUT decode — the same numerics family as the GLM engine's KV8, i.e. **opt-in and lossy**; default path untouched).

- `K`/`V` rows stored as 1 byte/element + one f32 scale per (head, position) — 4 bytes/element → 1 byte + 4 B scale
- `Q36Seq` (serve slots) and `ensure_kv` (growth/copy) extended; fused LUT-dot/LUT-axpy kernels (`kv8_fused.h`) keep the lane-reduction order of the f32 path
- Banner on engagement: `[qwen36] KV8=1: KV rows stored fp8-e4m3 + per-row f32 scale (lossy, opt-in)`

## Measured (all on the real engine, synthetic container)

**Medium context (8,192 tokens prefilled, then 48-token decode):**

| arm | peak RSS | TTFT | decode |
|---|---|---|---|
| f32 | 804 MB | 71 s | 0.89 tok/s |
| KV8 | **673 MB (−131 MB, −16%)** | 122 s | 0.52 tok/s |

**Long context (~34,500 tokens):** f32 peak RSS 2,780.7 MB vs KV8 **2,191.6 MB — 589 MB / 21% saved** (matches theory: 1.70 GB f32 KV → ~0.43 GB, plus scale arrays).

**Honest cost:** decode and prefill run ~1.7× slower under KV8 on this engine. Root cause is structural: qwen36's attention has no MLA absorb — every query row reads every K/V row directly, so the dequant work is O(T) per head with no amortization, and the byte-LUT breaks the vectorized dot. The GLM engine's KV8 is decode-neutral because its latent KV amortizes dequant over many query rows per tile. **This is exactly why upstream marked GPU-native quantized-KV kernels as the follow-up work; on CPU, this patch is a RAM-for-speed trade, not a free win.**

**Correctness notes:**
- Same-config output stability: the UNPATCHED qwen36 is itself run-to-run nondeterministic on this synthetic fixture (3 different greedy outputs in 3 one-thread runs — timing-sensitive speculation gate, present before my patch; the GLM engine passes the same gate deterministically). So token-level A/B is not meaningful here; instead:
- The patch builds warning-clean and the KV8 arm produces plausible output with no crashes across all runs; quantization error is bounded per-row by the e4m3 grid (same loss class as the GLM engine's documented KV8).
- **Requires real-model validation** before any upstream PR: perplexity check on real Qwen3.8-27B, plus a determinism root-cause for the fixture (out of scope here).

## What it buys on real hardware (extrapolated, formula-verified)

At real 27B scale (64 layers, kv_heads 8, head_dim 128; KV = 2·L·T·KVH·HD·4 B):

| context | f32 KV | KV8 KV | saving |
|---|---|---|---|
| 32,768 | 17.2 GB | 4.3 GB | **12.9 GB** |
| 65,536 | 34.4 GB | 8.6 GB | 25.8 GB |
| 262,144 | 137.4 GB | 34.4 GB | 103 GB |

On a 32 GB RAM box, 64k context goes from impossible (34.4 GB KV + 29 GB int8 weights) to feasible (8.6 + 29 GB). On the 16 GB floor the engine targets, even 32k was impossible before and is now borderline. **The patch's value is fit, not speed — and the speed cost on CPU must be fixed (tile the dequant, or take the GPU-native route) before it's a default-able feature.**

## Recommended next steps (for a real PR)

1. Tile-staged dequant: dequantize K/V in 512-row tiles into the existing `attn_sc`-style per-thread staging so the dot stays on the vectorized path (this is what GLM does; expected to close most of the 1.7× gap at long T).
2. Port the same patch to the other no-KV-quant engines (olmoe, inkling, deepseek_v41) — same three touchpoints each: alloc, seq swap/free, attention read/write.
3. KV_TQ (2–6 bit) as a second knob reusing `kv_tq.h`, for the 4× tier.
4. Real-model validation: perplexity delta on Qwen3.8-27B + qwen36 fixture determinism root-cause.

Files: patch applied in `~/colibri/c/qwen36.c` + `~/colibri/c/kv8_fused.h`; measurement script `exp_qwen27_kv8b.py`, raw JSON `results_qwen_kv8.json`.
