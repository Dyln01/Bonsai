# Research map — dense-model techniques for less RAM & more tok/s

**Date:** 2026-10-15 · survey of the literature, mapped against what colibri + our patches already have.

## 1. The technique families (papers → what they achieve → colibri status)

### A. Activation/contextual sparsity for dense MLPs — the family our HOT-NEURON work belongs to

| Paper | Core idea | Reported result | colibri/q36 status |
|---|---|---|---|
| **Deja Vu** (Liu et al., MLR 2023) | Contextual sparsity: ~85% of attention heads, ~95% of MLP can be skipped per input; a tiny 2-layer MLP predictor (4-10% of hidden size) predicts the active set from the previous layer's output, computed asynchronously | >2× vs FasterTransformer, >6× vs HF, no quality loss (OPT-175B) | **not present** — the biggest delta between our static prefix and the literature: their set is *per-token predicted*, ours is *per-corpus calibrated*. A predictor would make frac-cut quality workload-robust |
| **PowerInfer** (Song et al., SOSP'24) | Power-law neuron activation: ~26% of neurons carry 80% of activations ("hot neurons"); place hot on fast tier, cold computed elsewhere; online predictors gate the rest | up to 11.69× vs llama.cpp on RTX 4090, avg 7.23×; 13.2 tok/s avg on OPT-175B-class | **our HOT_FRAC is the static half of this** (their insight #1 = our measured 99.6%-energy-in-top-25%). Their dynamic half (predictor) = the Deja Vu delta above |
| **LLM in a Flash** (Alizadeh et al., Apple 2023/24) | Flash-resident weights + DRAM cache of *predicted-active* neurons; "windowing" (sliding reuse of recently active neurons, 10%→2.4% DRAM), "row-column bundling" (≥32 KiB contiguous reads), attention resident / FFN streamed | 2× DRAM-size models, 4-5× CPU speedup vs naive | **we have the storage half (TRUNK_MAP mmap) but not windowing** — a per-token sliding cache of active rows would cut our prefix further |
| **GRIFFIN** (Dong et al. 2024) | "Flocking": tokens in a sequence activate similar FF neurons; pick neurons from the *prompt itself*, once per sequence, no calibration at all | 50% FF sparsity ≈ no loss, 1.25-1.29× latency (Llama-2-13B, Gemma-7B) | **cheap add-on to our stack**: sequence-level selection could replace per-corpus calibration → better workload robustness with zero profile maintenance |
| **ShadowLLM** (EMNLP'24) | First layer's attention output predicts sparsity for the *whole* model (one predictor, not one per layer) | up to 20% faster than DejaVu | same slot as Deja Vu, cheaper predictor |
| **Polar Sparsity** (NeurIPS'25) | MLP sparsity vanishes under batching (union of activations), attention-head sparsity is batch-stable; selective GEMM/FlashAttention kernels | up to 2.2× batched decode, 5.5× selective-GEMM micro | relevant if we ever batch; single-stream decode is where our stack already plays |
| **CLAWS / CATS** (2025) | Calibration-aware sparsity for instruction-tuned models; dReLU-induced sparsity; custom ARM NEON scatter-gather kernel | ×1.24 wall-clock on full MLP block at 50% density (Apple silicon) | the "induce sparsity in SwiGLU models" route — needs a patch pass on the checkpoint (ReLUfication); our HOT_FRAC skips without touching weights |
| **ProSparse / ReLUfication** | post-train conversion of SwiGLU→ReLU to expose ~90% sparsity | 2-5× inference | same as above: weight-editing route, we deliberately avoid it (dial-not-delete) |

**Where our v3/v4 sits:** PowerInfer's static half + Apple's storage half + our own additions they
don't have (hotness-*sorted layout* making the active set a *contiguous prefix* → static working
set → page-cache-resident at ~100% hit, plus the quality dial). The literature's missing piece we
could still take: **the dynamic predictor** (Deja Vu/ShadowLLM) and **windowing** (Apple).

### B. Speculative decoding — multiplies any bandwidth saving

| Paper | Idea | Result | status |
|---|---|---|---|
| Leviathan/Chen 2023 (spec decoding) | small draft + parallel verify; each accepted token amortizes the whole weight sweep | 2-3× typical | q36 has **prompt-lookup drafts + S-row verify** (COLI_LOOKUP); a trained draft model would raise acceptance |
| **Ghidorah** (2025) | spec decode on edge Jetson + hetero-core scheduling | up to 7.6× decode | the scheduling part needs NUMA/big-little awareness; our box is symmetric |
| **DS2D** (Samsung, 2026) | tree-based self-speculation *without a draft model* via prefix-tuned multi-token heads | up to 2.3× | needs an MTP head — the 27B ships one (converter skips it today; noted in docs) |
| **ExSpec / Agent-X** (2026) | n-gram LUT as the draft; prompt rewriting for prefix-cache hits | 1.73× decode, 1.97× prefill | n-gram drafting ≈ our COLI_LOOKUP; **prompt rewriting for KV prefix reuse is absent and cheap** — kv_prefix.h exists per-session, not across agent turns |

### C. KV-cache attack (long-context RAM, not weight RAM)

| Paper | Idea | Result | status |
|---|---|---|---|
| KIVI (2024) | 2-bit per-channel KV | 2.35-3.47× throughput | our **KV8** = 8-bit fp8 (4× vs f32); KV_TQ ships 4-6-bit on GLM only — porting KV_TQ to q36 is the remaining 2× |
| KVQuant (2024) | hierarchical KV quant | 10M-token contexts | same slot |
| ThinK (2024) | channel-pruned KV | O(√n) attention memory | not present; complements KV8 |
| **OasisKV** (2026) | KV blocks off HBM; speculative drafts predict which blocks the *next* step needs → prefetch | decode beyond HBM at ~full speed | for q36 the analog is: drafts predict which KV *rows* matter → skip the O(T) scan; matches our flagged "absorbed-score/MLA" next lever |

### D. Storage/compute-in-memory (next frontier, hardware-bound)

- **Routide / "Paging the Experts"** (2026): flash-backed MoE on iPhone — the MoE analog of what we built for dense; validates the storage-backed approach on real phones.
- **Compute-in-Flash LLM inference** (2026): integer-only pipelines + dictionary KV executed *inside* the SSD controller — 3-4× latency/energy savings; needs CSD hardware we don't have, but the integer-only trick (avoiding fp dequant like our map kernels do) is portable.
- **Delta prefetching** (2026, "storage-aware sparse inference"): fetch only *diffs* of neuron sets between consecutive tokens — direct extension of Apple's windowing for our paged path.

## 2. Ranked: what to actually implement next in colibri/q36

Ranked by (impact on our measured bottlenecks) × (no-download implementability) × (fit with existing patch seams):

1. **GRIFFIN-style sequence-level selection** (days, no training, no calibration)
   Use the prompt itself to pick the neuron set per sequence — add to our paged kernel as a
   gather-list instead of a prefix cut. Fixes hot-neuron's one weakness (workload drift)
   with zero profile maintenance. Literature says 50% sparsity ≈ lossless on many models.
2. **KV_TQ (2-4 bit) port to q36** (days) — halves KV8 again; the KV8 patch structure
   (kv8_fused.h + parity gates) is the template; KIVI per-channel is the better scheme.
3. **Sliding-window neuron cache** (Apple's windowing, ~1 week) — per-token active-set cache
   above the prefix: prefix covers the corpus-hot rows, window covers the token-hot tail.
   Literature: DRAM fraction 10% → 2.4%.
4. **Draft-model speculation with MTP head** (1-2 weeks) — the 27B ships an MTP head the
   converter currently skips; wiring it into q36_spec_step multiplies every bandwidth win 2-3×.
5. **Predictor-based dynamic sparsity** (Deja Vu/ShadowLLM, 2-3 weeks) — a 4-10%-width 2-layer
   predictor per layer, trained on our own synthetic activation dumps (HOT_PROFILE already
   collects exactly this data — the training set is free).
6. **Prefix-cache prompt structuring** (days, agent-side) — kv_prefix reuse across tool-call
   turns; the engine's `.coli_usage`+kv_prefix seams exist.

## 3. What the literature confirms about our measured design choices

- The 99.6%-energy-in-top-25% we measured on synthetics matches PowerInfer's 80%-compute-in-26%
  and Deja Vu's 95% MLP sparsity — three independent measurements of the same power law.
- Apple's "attention resident, FFN streamed" split is exactly our TRUNK_MAP layout choice.
- The survey consensus ordering — quantize first, then paging, then KV compression — is the
  order we built in (int4 container → mapped trunk → KV8 → hot-neuron).
- Weight-only dequant "effectively free during bandwidth-bound decode" (the I/O survey) is
  precisely our MAPFAST lesson — the dequant must be amortized (staged per row), not per element.
