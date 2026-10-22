# Cross-engine KV8/KV_TQ ports (Phase 4c) — design contract

**Date:** 2026-10-22 · status: **slice 1 shipped (build proof + CI-sized qwen38 container)**
**Order:** qwen38 first (GQA sibling: same `st.h`, spec machinery, chain idioms,
and an in-repo generator), glm53 second (MLA: latent KV makes TQ absorption
intrinsic — the codec rides ~1/8 the rows of a GQA cache).

## What travels unchanged

- `engine/kv_tq.h` codecs (PolarQuant / rotated-int4) — written engine-agnostic;
  the 3b-rot absorbed-rotation pattern (rotate queries once, dot raw rotated
  levels, accumulate rotated, unrotate once) applies verbatim to any GQA decode.
- `engine/kv8_fused.h` + the scale-hoisting contract (radius/scale in the same
  per-row slot, `dequant(r=1)*s == dequant(s)`).
- The parity gates (`kv8_parity.c`, `kv_tq_parity_q36.c`, `tq_absorb_parity.c`)
  compile against any engine that overlays the headers — codec math is engine-free.
- Methodology: `DUMP_IDS`/`DUMP` identity and cosine gates, honest `[timers]`
  decode numbers, `run_battery.sh` (its `Q36_ENGINE` env already parameterizes
  the binary; per-engine prompt/cap args documented per slice).

## What is engine-specific (per-engine patchers, own manifest section)

KV pool layout and growth path, decode-attention anchors (qwen38: GQA like
qwen36; glm53: MLA latent + decode-absorb already), knob names/parse sites,
flash/prefill tile loops, report lines. Each gets its own patcher
(`tools/patch_kv8_q38.py`, `tools/patch_tq_q38.py`, `tools/patch_kvq_glm53.py`),
its own backup suffix (`.q38prekv8` …), and a **separate `cross_engine` section
in `tools/patch_manifest.json`** — never interleaved with the qwen36 chain,
which stays the source of truth for this repo's engine snapshot.

Upstream drift policy: patchers anchor to colibri **bf24429** sources; the CI
build-proof step (slice 1) recompiles both siblings on every push, so an
upstream anchor break surfaces as a red build within minutes of a pin change,
and pins move only by explicit commit.

## Gates per slice

1. **Slice 1 (this commit):** `make qwen38 glm53` green in CI with Bonsai
   headers overlaid; `gen_qwen38_27b_f16.py` SYNTH_-parameterized small
   container generated in CI. No tier code yet.
2. **Slice 3 (qwen38 KV_TQ + absorb): shipped, gates as above.**
- **Slice 2 (qwen38 KV8):** codec parity unchanged; kv8-off last-logits
   byte-identical to stock qwen38; kv8-on self-deterministic across runs;
   kv8-on vs off **prefill-only** cosine ≥ 0.95 (e4m3 noise band on synthetic
   random nets — a first 0.999 attempt was a gate-design error: decode on a
   chaotic synthetic net amplifies honest quantization noise into sequence
   divergence, cosine ~0.09 with a CORRECT implementation; qwen36 never gated
   lossy tiers this way for exactly this reason).
3. **Slice 3 (qwen38 KV_TQ + absorbed rotation):** same identity/cosine ladder
   per tier (tq4/tq3), plus `Q38_TQ_ABSORB` 0/1 id-identity (the absorb gate
   pattern from qwen36 3b), plus the sim/battery arms on the mini-PC.
4. **Slice 4 (glm53 latent TQ):** MLA latents are already absorbed; TQ on the
   latent rows must be decode-neutral-or-better (that is the GLM engine's
   documented KV8 property) — gate: cosine band + decode ms/tok non-regression
   at equal RSS, on a glm53 synthetic container (generator lands with the slice).
5. Every slice: qwen36 chain + all existing gates stay green (no cross-talk).

## Out of scope

deepseek-v4 engines, Vulkan/CUDA KV tiers, serve-mux KV-slot semantics per
engine (qwen36-only feature), real-weights runs (Phase 5 covers qwen36 first;
siblings get batteries only if their ports pass synthetic gates).

## Slices

- [x] 1 — build proof + parameterized qwen38 generator (CI green)
- [x] 2 — `tools/patch_kv8_q38.py` (12 anchored edits over qwen38.c +
      qwen38_core.h: knob, byte-wide pools + scales, quantize-on-write,
      dequant-once-per-selected-row in the QSA gather, prefix/serve refusals)
      + CI **compile gate** (stock build, patch, rebuild). Behavioral gates
      blocked on 2a: the in-repo qwen38 generator emits the qwen3_5 dialect while
      pinned qwen38 requires `qwen4_exp_text` (+ PLE/QSA config surface).
- [ ] 2a — qwen38-dialect synthetic container: reverse-engineer the
      qwen38_core.h parser (config: model_type/hidden_act/output_gate_type/
      attention_bias/tie_word_embeddings/rope_parameters + PLE & QSA indexer
      numerics + tensor naming incl. ple_* weights) and fix
      `gen_qwen38_27b_f16.py` (SYNTH_-parameterized already).
- [ ] 2b — behavioral gates: stock vs patched-KV8=0 `DUMP_IDS` identity;
      KV8=1 vs 0 last-logit cosine ≥ 0.999; RSS delta = row-width math.
- [x] 3 — `patch_tq_q38.py` (KV_TQ tiers + absorbed rotation on qwen38):
      CI gates green — absorb 0-vs-1 cosine **1.0** (prefill and full), absorb1
      self-deterministic byte-identical, tq-on-vs-off prefill cosine 0.91 inside the
      Lloyd-Max band (>=0.85 pre-registered). Serve/mux refused under KV_TQ as under KV8.
- [x] 4a — `scripts/gen_glm53_synth.py` (glm53 dialect per Appendix B; stock
      engine loads + runs it first try; CI load-smoke green).
- [ ] 4b — `patch_kvq_glm53.py`: codec-on-latent TQ port + gates (off-identity,
      self-determinism, prefill noise band). Absorption-native: MLA decode dots
      queries against latents through absorbed kv_b weights, so no query
      rotation is needed.

## Appendix A — qwen38 container spec (extracted from colibri bf24429 `qwen38_core.h`)

Checklist for slice 2a-2 (generator rewrite). `<P>` = `model` or
`model.language_model` (auto-detected via the embed name). All weights load via
`q38_load_weight` (quantizes per engine policy) or `q38_load_tensor` (resident f32).

**config.json (text_config or root):** `model_type="qwen4_exp_text"`,
`hidden_act="silu"`, `output_gate_type="sigmoid"`, `attention_bias=false`,
`tie_word_embeddings=false`, `hidden_size`, `num_hidden_layers`, `vocab_size`,
`max_position_embeddings`, `eos_token_id`, `rms_norm_eps`,
`rope_parameters{rope_type="default", rope_theta, partial_rotary_factor}`,
`hc_count` (def 4), `hc_lowrank` (def 320; hc_width = hc_count*hidden),
`num_attention_heads`, `num_key_value_heads`, `head_dim`,
`indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim`, `indexer_budget`,
`indexer_compress_ratio`, `num_experts`, `num_experts_per_tok`,
`moe_intermediate_size`, `shared_expert_intermediate_size`, `norm_topk_prob`,
`linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`,
`linear_value_head_dim`, `linear_conv_kernel_dim`,
`ple_embed_dim` (def hidden), `ple_conv_kernel_size` (def 4), `ngram_size` (def 3),
`heads_per_ngram` (def 8), `split_ngram_parts` (def 1),
`ple_layer_ids` (REQUIRED by the pinned engine: exactly one layer), `layer_types` (REQUIRED,
one per layer: `full_attention`|`qwen_sparse_attention`|`linear_attention`).

**Boundaries:** `<P>.embed_tokens.weight` [V,H] · `lm_head.weight` [V,H]
(no prefix) · `<P>.hyper_connection_mixer.{hc_norm.weight [hc_width],
input_mix_weight_down.weight [rank,width], input_mix_weight_up.weight [width,rank]}`
(no block_inject on the final mixer).

**Every layer:** `attn_hyper_connection.*` and `mlp_hyper_connection.*` =
`{hc_norm.weight, input_mix_weight_down.weight, input_mix_weight_up.weight,
block_inject_weight.weight [hc_count,hc_width]}` · `mlp.gate.weight` [E,H] ·
`mlp.shared_expert.{gate,up,down}_proj.weight` · `mlp.shared_expert_gate.weight` [H] ·
experts: per-expert `mlp.experts.<e>.{gate,up,down}_proj.weight` (a fused
`mlp.experts.gate_up_proj` variant also exists; use the per-expert form).

**full_attention layers:** `self_attn.{q_proj [QH*HD*2,H], k_proj [KVH*HD,H],
v_proj [KVH*HD,H], o_proj [H,QH*HD]}` · `self_attn.{q_norm,k_norm}.weight` [HD] ·
`self_attn.indexer.index_qk_proj.weight` [(IQH+IKH)*ID, H] ·
`self_attn.indexer.{q_layernorm,k_layernorm}.weight` [ID].

**linear_attention layers:** `linear_attn.{in_proj_qkv [conv_dim,H], in_proj_z [vd,H],
in_proj_b [vh,H], in_proj_a [vh,H], conv1d.weight [conv_dim*convk], dt_bias [vh],
A_log [vh], norm.weight [vdim], out_proj.weight [H,vd]}`.

**PLE layer (the one named in `ple_layer_ids`):** `ple.key_proj.weight`
[hc_width, ple_dim], `ple.value_proj.weight` [hidden, ple_dim],
`ple.{norm_key,norm_query,norm_conv}.weight` [hc_width], `ple.conv1d.weight`
[hc_width, ple_conv_kernel_size], `ple.ple_embedding.layer_multipliers` (I64,
ngram_size), `ple.ple_embedding.ngram_heads_vocab_sizes` / `..._offsets` (I64,
ngram_heads), `ple.ple_embedding.ngram_embedding.weight` [rows >= max(off+vocab),
ple_dim/ngram_heads].

**Optional:** vision only if
`Q38_VISION=1` (set 0 in CI); MTP head only if the container carries it.

## Appendix B — glm53 container spec (extracted from colibri bf24429 `glm53.c`)

Checklist for slice 4 (generator + latent-TQ port). MLA engine: the KV cache
holds **latent rows** (`kv_lora_rank` + `qk_rope_head_dim` floats), `kv_b_proj`
is absorbed at load (`absorb_kvb`), so the TQ port target is the latent row —
codec-on-latent is absorption-native by construction.

**config.json (text_config or root), required:** `hidden_size`,
`num_hidden_layers`, `vocab_size`, `intermediate_size`, `num_attention_heads`,
`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`,
`v_head_dim`, `n_routed_experts`, `num_experts_per_tok`,
`moe_intermediate_size`, `n_shared_experts`, `routed_scaling_factor`,
`rms_norm_eps`, `swiglu_limit`, `num_nextn_predict_layers` (0 = no MTP block),
`hc_mult`, `hc_sinkhorn_iters`, `hc_eps`, plus `linear_attn_config{}` with
`num_heads`, `head_dim`, `short_conv_kernel_size`, `gate_lower_bound`,
`index_topk`, `index_n_heads`, `index_head_dim`, `index_kpool`,
`index_kpool_always_select_tail`; `layer_types` (per layer, contains
"linear" or "attention"/"full"), `mlp_layer_types`, `first_k_dense_replace`,
`sparse`. Vision: `vision_config` ABSENT = no vision tensors.

**Prefix:** `model.language_model.` else `model.` (probed via embed name).
**Head is TIED to the embedding** ("the embedding (and a head tied to it)") —
no `lm_head` tensor. `enorm/eh_proj/e_proj`-style MTP tensors only when
`num_nextn_predict_layers > 0`.

**Every text layer:** `input_layernorm.weight`, `post_attention_layernorm.weight`,
hyper-connection triplets `hc_attn_{base,fn,scale}` and `hc_ffn_{base,fn,scale}`
(shapes from `hc_mult`; finalize at generator time), then per kind:
- **MLA (full) layer:** `self_attn.{q_a_proj,q_b_proj}.weight`,
  `self_attn.q_a_layernorm.weight`, `self_attn.kv_a_proj_with_mqa.weight`,
  `self_attn.kv_a_layernorm.weight`, `self_attn.kv_b_proj.weight`,
  `self_attn.o_proj.weight`, indexer: `self_attn.indexer.{wq_b,wk,weights_proj}.weight`,
  `self_attn.indexer.k_norm.{weight,bias}`
- **linear layer:** `self_attn.{f_a_proj,f_b_proj,b_proj}.weight`,
  `self_attn.{A_log,dt_bias}` (conv naming finalize at generator time)
- **MoE MLP:** `mlp.gate.weight`, `mlp.gate.e_score_correction_bias`,
  `mlp.experts.<e>.{gate,up,down}_proj.weight` (int4-gs64 sidecar `.qs`
  accepted), `mlp.shared_experts.{gate,up,down}_proj.weight` (plural)
- **dense MLP (i < first_k_dense_replace):** `mlp.{gate,up,down}_proj.weight`

**Slice-4 port target:** quantize the latent KV row (`kv_lora_rank +
qk_rope_head_dim` floats, written post `kv_a_layernorm` + rope-split) with
`coli_kvq_*`; MLA decode already dots queries against latents through the
absorbed `kv_b` weights, so TQ-on-latent needs no query rotation — the
absorb is the architecture. Gates mirror slice 3 (off-identity,
self-determinism, prefill noise band) on a SYNTH_-sized glm53 container.
