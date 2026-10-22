#!/usr/bin/env python3
"""gen_qwen38_27b_f16.py — SYNTHETIC qwen38 container in the qwen4_exp_text dialect.

Rewritten 2026-10-22 (Phase 4c slice 2a-2) against Appendix A of
docs/CROSS_ENGINE_KVQ.md, extracted from colibri bf24429 qwen38_core.h:
hyper-connection mixers (per attn/mlp per layer + boundary), QSA indexer on
full-attention layers, DeltaNet nine-tensor set on linear layers, per-expert
MoE layout, strict config keys, layer_types array. PLE/vision/MTP absent by
construction (ple_layer_ids omitted; CI runs with Q38_VISION=0).

Defaults = the real 27B-class geometry (huge on disk); SYNTH_* overrides give
CI-sized containers with the same code paths:

  SYNTH_OUT SYNTH_H SYNTH_L SYNTH_MLP SYNTH_V SYNTH_QH SYNTH_KVH SYNTH_HD
  SYNTH_E SYNTH_TOPK SYNTH_MI SYNTH_SI SYNTH_IQH SYNTH_IKH SYNTH_ID

All tensors F16; single shard; no real weights, seeded rng.
"""
import json
import os
import struct
import sys

import numpy as np

OUT = os.path.expanduser(os.environ.get("SYNTH_OUT", "~/synth_models/synth_qwen38_27b_f16"))
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

H   = int(os.environ.get("SYNTH_H",   5120))
L   = int(os.environ.get("SYNTH_L",   64))
MLP = int(os.environ.get("SYNTH_MLP", 17408))   # routed expert inter
V   = int(os.environ.get("SYNTH_V",   248320))
QH  = int(os.environ.get("SYNTH_QH",  40))
KVH = int(os.environ.get("SYNTH_KVH", 8))
HD  = int(os.environ.get("SYNTH_HD",  128))
E   = int(os.environ.get("SYNTH_E",   32))
TOPK = int(os.environ.get("SYNTH_TOPK", 4))
SI  = int(os.environ.get("SYNTH_SI",  512))     # shared expert inter
IQH = int(os.environ.get("SYNTH_IQH", 4))       # indexer q heads
IKH = int(os.environ.get("SYNTH_IKH", 1))       # indexer kv heads
ID  = int(os.environ.get("SYNTH_ID",  64))      # indexer head dim
HC_COUNT, HC_RANK = 2, 64
HCW = HC_COUNT * H
DN_KH, DN_VH, DN_KD, DN_VD, CONVK = 4, 8, 64, 64, 4
CONV_DIM = 2 * DN_KH * DN_KD + DN_VH * DN_VD
VD = DN_VH * DN_VD
IBUDGET, IRATIO = 2048, 4

rng = np.random.default_rng(27)

def save_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        b = arr.tobytes()
        dt = {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32",
              np.dtype(np.int64): "I64"}[arr.dtype]
        header[name] = {"dtype": dt,
                        "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for b in blobs: f.write(b)

def f16mat(o, i, s=None):
    if s is None: s = (2.0 / (o + i)) ** 0.5 * 0.5
    return (rng.standard_normal((o, i)) * s).astype(np.float16)

def f16vec(n):
    return (np.ones(n, np.float32) * 0.6 + rng.random(n).astype(np.float32) * 0.3).astype(np.float16)

layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(L)]
PLE_LAYER = 1 if L > 2 else 0          # qwen38 REQUIRES exactly one PLE layer
NGRAM_HEADS = (3 - 1) * 8               # (ngram_size-1) * heads_per_ngram
NGRAM_VOCAB, NGRAM_DIM = 64, H // NGRAM_HEADS
cfg = {
    "architectures": ["Qwen4ExpTextForCausalLM"],
    "model_type": "qwen4_exp_text",
    "hidden_act": "silu",
    "output_gate_type": "sigmoid",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "hidden_size": H, "num_hidden_layers": L, "vocab_size": V,
    "max_position_embeddings": 32768, "eos_token_id": V - 1, "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.5},
    "hc_count": HC_COUNT, "hc_lowrank": HC_RANK,
    "num_attention_heads": QH, "num_key_value_heads": KVH, "head_dim": HD,
    "indexer_n_heads": IQH, "indexer_kv_heads": IKH, "indexer_head_dim": ID,
    "indexer_budget": IBUDGET, "indexer_compress_ratio": IRATIO,
    "num_experts": E, "num_experts_per_tok": TOPK,
    "moe_intermediate_size": MLP, "shared_expert_intermediate_size": SI,
    "norm_topk_prob": True,
    "linear_num_key_heads": DN_KH, "linear_num_value_heads": DN_VH,
    "linear_key_head_dim": DN_KD, "linear_value_head_dim": DN_VD,
    "linear_conv_kernel_dim": CONVK,
    "ngram_size": 3, "heads_per_ngram": 8, "split_ngram_parts": 1,
    "ple_layer_ids": [PLE_LAYER + 1],
    "layer_types": layer_types,
}
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)

t = {}
t["model.embed_tokens.weight"] = f16mat(V, H, 0.02)
t["lm_head.weight"] = f16mat(V, H, 0.02)
for base in ("model.hyper_connection_mixer",):
    t[base + ".hc_norm.weight"] = f16vec(HCW)
    t[base + ".input_mix_weight_down.weight"] = f16mat(HC_RANK, HCW)
    t[base + ".input_mix_weight_up.weight"] = f16mat(HCW, HC_RANK)

def gr(prefix):
    t[prefix + ".hc_norm.weight"] = f16vec(HCW)
    t[prefix + ".input_mix_weight_down.weight"] = f16mat(HC_RANK, HCW)
    t[prefix + ".input_mix_weight_up.weight"] = f16mat(HCW, HC_RANK)
    t[prefix + ".block_inject_weight.weight"] = f16mat(HC_COUNT, HCW)

for li in range(L):
    p = f"model.layers.{li}."
    gr(p + "attn_hyper_connection")
    gr(p + "mlp_hyper_connection")
    t[p + "mlp.gate.weight"] = f16mat(E, H)
    t[p + "mlp.shared_expert.gate_proj.weight"] = f16mat(SI, H)
    t[p + "mlp.shared_expert.up_proj.weight"] = f16mat(SI, H)
    t[p + "mlp.shared_expert.down_proj.weight"] = f16mat(H, SI)
    t[p + "mlp.shared_expert_gate.weight"] = f16mat(1, H)
    for e in range(E):
        t[p + f"mlp.experts.{e}.gate_proj.weight"] = f16mat(MLP, H)
        t[p + f"mlp.experts.{e}.up_proj.weight"] = f16mat(MLP, H)
        t[p + f"mlp.experts.{e}.down_proj.weight"] = f16mat(H, MLP)
    if layer_types[li] != "linear_attention":
        t[p + "self_attn.q_proj.weight"] = f16mat(QH * HD * 2, H)
        t[p + "self_attn.k_proj.weight"] = f16mat(KVH * HD, H)
        t[p + "self_attn.v_proj.weight"] = f16mat(KVH * HD, H)
        t[p + "self_attn.o_proj.weight"] = f16mat(H, QH * HD)
        t[p + "self_attn.q_norm.weight"] = f16vec(HD)
        t[p + "self_attn.k_norm.weight"] = f16vec(HD)
        t[p + "self_attn.indexer.index_qk_proj.weight"] = f16mat((IQH + IKH) * ID, H)
        t[p + "self_attn.indexer.q_layernorm.weight"] = f16vec(ID)
        t[p + "self_attn.indexer.k_layernorm.weight"] = f16vec(ID)
    else:
        t[p + "linear_attn.in_proj_qkv.weight"] = f16mat(CONV_DIM, H)
        t[p + "linear_attn.in_proj_z.weight"] = f16mat(VD, H)
        t[p + "linear_attn.in_proj_b.weight"] = f16mat(DN_VH, H)
        t[p + "linear_attn.in_proj_a.weight"] = f16mat(DN_VH, H)
        t[p + "linear_attn.conv1d.weight"] = f16mat(CONV_DIM, CONVK)
        t[p + "linear_attn.dt_bias"] = f16vec(DN_VH)
        t[p + "linear_attn.A_log"] = (-rng.random(DN_VH).astype(np.float32) * 2).astype(np.float16)
        t[p + "linear_attn.norm.weight"] = f16vec(DN_VD)
        t[p + "linear_attn.out_proj.weight"] = f16mat(H, VD)
    if li == PLE_LAYER:
        pp = p + "ple."
        t[pp + "key_proj.weight"] = f16mat(HCW, H)          # ple_dim defaults to hidden
        t[pp + "value_proj.weight"] = f16mat(H, H)
        t[pp + "norm_key.weight"] = f16vec(HCW)
        t[pp + "norm_query.weight"] = f16vec(HCW)
        t[pp + "norm_conv.weight"] = f16vec(HCW)
        t[pp + "conv1d.weight"] = f16mat(HCW, 4)
        t[pp + "ple_embedding.layer_multipliers"] = np.array([1, 2, 3], dtype=np.int64)
        t[pp + "ple_embedding.ngram_heads_vocab_sizes"] = np.full(NGRAM_HEADS, NGRAM_VOCAB, dtype=np.int64)
        offs = np.arange(NGRAM_HEADS, dtype=np.int64) * NGRAM_VOCAB
        t[pp + "ple_embedding.ngram_heads_offsets"] = offs
        t[pp + "ple_embedding.ngram_embedding.weight"] = f16mat(NGRAM_HEADS * NGRAM_VOCAB, NGRAM_DIM)
    if li % 8 == 0:
        print(f"layer {li} done", flush=True)

save_st(f"{OUT}/model-00000.safetensors", t)
mb = os.path.getsize(f"{OUT}/model-00000.safetensors") / 1e6
print(f"container: {mb:.1f} MB, {L} layers ({layer_types.count('full_attention')} attn + "
      f"{layer_types.count('linear_attention')} linear) x {E} experts, qwen4_exp_text dialect")
