# MTP head for qwen36 — design contract (Phase 2, pre-implementation)

**Date:** 2026-10-16 · status: **design pinned, implementation next**
**Rule honored:** no checkpoint downloaded. The contract below is pinned from
(a) the checkpoint's `model.safetensors.index.json` (a 98 KB name→shard map,
no weights), (b) its `config.json`, (c) three independent implementations
that cross-attest the head math (ROCm/ATOM's `qwen3_5_mtp.py` notes, the
`quivent/qwen-mtp-tensors` GGUF archaeology for the dense 27B, vLLM's
`Qwen3_5MultiTokenPredictor` as described in the vLLM-Ascend MTP writeup),
and (d) the in-tree reference: qwen38's `Q38_MTP` (`qwen38_core.h`,
`docs/speculative.md`).

## What the checkpoint carries (Qwen/Qwen3.6-35B-A3B — authoritative)

config: `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`
(shared embed + shared lm_head). 19 tensors, prefix `mtp.` at the checkpoint
root (the model body sits under `model.language_model.`):

| tensor | role | shape basis |
|---|---|---|
| `mtp.fc.weight` | concat projection `[hidden, 2*hidden]` | the "eh_proj" |
| `mtp.pre_fc_norm_embedding.weight` | rmsnorm of the token-embedding half | `[hidden]` |
| `mtp.pre_fc_norm_hidden.weight` | rmsnorm of the hidden-state half | `[hidden]` |
| `mtp.layers.0.input_layernorm.weight` | head layer pre-attn norm | `[hidden]` |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight` | **full attention only** (no DeltaNet in the head) | q/kv head config |
| `mtp.layers.0.self_attn.{q,k}_norm.weight` | qk-norm (same as model layers) | `[head_dim]` |
| `mtp.layers.0.post_attention_layernorm.weight` | pre-MLP norm | `[hidden]` |
| `mtp.layers.0.mlp.gate.weight` | router (MoE head, 35B) | `[E, hidden]` |
| `mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight` | shared expert | dense-27B variant: plain `mlp.{gate,up,down}_proj` |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | shared-expert gate | `[1, hidden]` |
| `mtp.layers.0.mlp.experts.{gate_up,down}_proj` | fused experts (same layout as model layers) | |
| `mtp.norm.weight` | final norm before the shared lm_head | `[hidden]` |

The head layer's name set is **exactly a model layer's** (input/post norms,
qk-normed GQA, MoE block) plus the three mtp-specific tensors — so it can
load through the same per-layer loader path as layer index `n_layers`.

## The head math (three implementations agree)

```
e   = embed(t_p)                                # shared embedding, the just-picked token
h   = model_hidden[p]                           # the model's final hidden at the last settled position
x   = fc( concat( rmsnorm(e, pre_fc_norm_embedding),
                  rmsnorm(h, pre_fc_norm_hidden) ) )
y   = head_layer(x)                             # ONE full-attention layer (own KV rows)
d1  = argmax( lm_head( rmsnorm(y, mtp.norm) ) )  # draft 1
# deeper drafts feed the head's own hidden back in:
x'  = fc( concat( rmsnorm(embed(d1), pre_fc_norm_embedding),
                  rmsnorm(y,     pre_fc_norm_hidden) ) )
d2  = argmax( lm_head( rmsnorm(head_layer(x'), mtp.norm) ) )   # etc.
```

- Draft 1 reads the model's hidden state; every deeper draft reads the head's
  own previous-row hidden (matching qwen38's "head's own streams" wording).
- The head keeps its own KV rows (one per settled position, plus a stale tail
  past the standing rows on a rejected verify — same overwrite-tail treatment
  as everywhere in these engines; the head has NO recurrent state, so its
  rollback is simpler than the model's).
- Attention: full attention with the model's RoPE (partial factor 0.25,
  theta 1e7; `mtp.rope_theta` may override — q38 precedent).

## q36 implementation plan (mirrors Q38_MTP; docs/speculative.md is the guide)

1. **Container**: converter stops skipping `mtp.*` — writes the head as
   **layer `n_layers`** in the per-layer shard series (its names map 1:1),
   the three mtp-specific tensors (`fc`, two pre-norms, final norm) into the
   globals shard, and `"mtp": 1` into `qwen36_meta.json`. A dense model's
   head (plain MLP) classifies by the same name rules. `qwen36_tensor_kinds.py`
   grows the mtp entries (classified, never silently dropped).
2. **Loader**: when `meta.mtp`, load layer `n_layers` + the mtp globals; the
   head's K/V pool comes from the same `[n_layers+1]` allocation (q38's
   `q38_layer_rows` trick), its experts from the same per-layer expert slots.
   The head is NOT in the model's forward loop — it runs only when drafting.
3. **Draft source**: in `q36_spec_step`, after the caller picks a token:
   capture the model's final hidden (small hook where `step_ex` runs
   `lm_head`), run the head once per draft, propose up to
   `Q36_MTP_DRAFTS` tokens as `SPEC_SRC_MTP`. The gate (`spec_draft.h`)
   already models this source: priors 0.8/0.6/0.5, per-draft cost `d`,
   `spec_gate_result(SPEC_SRC_MTP, …)`.
4. **Source choice**: with both MTP and lookup active, the higher expected
   tokens-per-second proposal wins (q38 rule; the gate exposes the pieces).
5. **Knobs**: `Q36_MTP=1` (opt-in initially; refuses a headless container),
   `Q36_MTP_DRAFTS=1|2|3|0` (0 = gate picks, default 2 per q38's measured
   optimum), `Q36_MTP_FORCE=accept|mixed|cycle|rowN` (test oracle, the
   pattern `Q38_MTP_FORCE`/`COLI_LOOKUP_FORCE` already use),
   `Q36_MTP_DUMP=<file>` if needed. Report line: `[qwen36 MTP] run: …`
   alongside `[qwen36 lookup]`.
6. **Synthetic container**: the generator emits a random-weight head in the
   same contract (so the whole path runs with zero downloads); acceptance
   rates on random weights are meaningless, so the gates are:
   - `Q36_MTP_FORCE=accept` end-to-end reproduces the oracle text exactly
     (the correctness proof of verify + head-row rollback);
   - `mixed`/`cycle` exercise rejections + rollback;
   - drafts-off vs drafts-on with the head disabled is byte-identical
     (no code-path change without the knob);
   - measured: head forward cost per draft (the gate's `d`), verify `T(n)`
     by row count, tokens/forward under forced acceptance.
7. **Honest label**: the head math is pinned from three independent
   implementations + the tensor index, but no real weights were run — the
   real-checkpoint acceptance rate and quality neutrality are Phase 5 items
   (gated on user sign-off), exactly like KV_TQ and the prefix-reuse
   reply-inclusive case.

## Why this is the big lever

Speculation multiplies every bandwidth win already built: each accepted
draft amortizes the model's full weight sweep (dense int4 trunk, expert
reads, attention) over one more token. q38's measured analog: GLM-5.2's
int8 MTP head drafts 2.2–2.8 tokens/forward when it pays; DS2D-style
self-speculation (no separate draft model to download) reports up to 2.3×.
On the mini-PC target (RAM-bound, 3.4–12 tok/s decode), a 2× decode
multiplier is worth more than every remaining RAM optimization combined.

## References

- in-tree: `docs/speculative.md` (the wiring guide), `spec_draft.h`
  (engine-agnostic gate + `SPEC_SRC_MTP`), `qwen38_core.h` (`q38_mtp_attach`
  and the head's stream handling), `qwen36.c` L5514+ (`Q36Spec`, the state
  machine being extended).
- external (code, no weights): `ROCm/ATOM` `docs/model_support_guide.md`
  (head design description), `quivent/qwen-mtp-tensors` (GGUF tensor map for
  the dense 27B head), vLLM `Qwen3_5MultiTokenPredictor` (step indexing
  `layers[spec_step_idx % num_mtp_layers]`), the checkpoint's
  `model.safetensors.index.json` + `config.json`.
