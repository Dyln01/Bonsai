#!/usr/bin/env python3
"""gen_glm53_synth.py — SYNTHETIC glm53 container (qwen4/glm5.3 MLA dialect).

Written 2026-10-22 (Phase 4c slice 4a) against docs/CROSS_ENGINE_KVQ.md
Appendix B, extracted from colibri bf24429 glm53.c: MLA latents (kv_b absorbed
at load), KDA linear-attention layers with q/k/v short convs, DSA lightning
indexer with k-pooling tensors, mHC hyper-connection triplets at both sites,
tied head (NO lm_head tensor), optional vision/MTP absent by construction.

Defaults are CI-sized; SYNTH_* overrides scale up:
  SYNTH_OUT SYNTH_H SYNTH_L SYNTH_V SYNTH_NH SYNTH_QL SYNTH_KL SYNTH_QK
  SYNTH_VH SYNTH_E SYNTH_TOPK SYNTH_MI SYNTH_DI SYNTH_FD SYNTH_KH SYNTH_KD
  SYNTH_CK SYNTH_IH SYNTH_ID SYNTH_IT SYNTH_IP SYNTH_HC

Experts ship int4-gs64 (U8 + .qs) like the coder30 containers; everything else
F16. No real weights, seeded rng, nothing downloaded.
"""
import json
import os
import struct
import sys

import numpy as np

OUT = os.path.expanduser(os.environ.get("SYNTH_OUT", "~/synth_models/synth_glm53_ci"))
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

H   = int(os.environ.get("SYNTH_H",   256))
L   = int(os.environ.get("SYNTH_L",   4))
V   = int(os.environ.get("SYNTH_V",   2048))
NH  = int(os.environ.get("SYNTH_NH",  4))     # MLA attention heads
QL  = int(os.environ.get("SYNTH_QL",  64))    # q lora rank
KL  = int(os.environ.get("SYNTH_KL",  32))    # kv lora rank (latent width)
QK  = int(os.environ.get("SYNTH_QK",  32))    # qk_nope head dim (qk_rope = 0: NoPE)
VH  = int(os.environ.get("SYNTH_VH",  32))    # v head dim
E   = int(os.environ.get("SYNTH_E",   4))
TOPK = int(os.environ.get("SYNTH_TOPK", 2))
MI  = int(os.environ.get("SYNTH_MI",  128))   # routed expert inter
DI  = int(os.environ.get("SYNTH_DI",  256))   # dense inter
FD  = int(os.environ.get("SYNTH_FD",  2))     # first_k_dense_replace
KH  = int(os.environ.get("SYNTH_KH",  4))     # KDA heads
KD  = int(os.environ.get("SYNTH_KD",  32))    # KDA head dim
CK  = int(os.environ.get("SYNTH_CK",  4))     # short conv kernel
IH  = int(os.environ.get("SYNTH_IH",  2))     # indexer heads
ID  = int(os.environ.get("SYNTH_ID",  16))    # indexer head dim
IT  = int(os.environ.get("SYNTH_IT",  8))     # index topk
IP  = int(os.environ.get("SYNTH_IP",  2))     # index kpool
HC  = int(os.environ.get("SYNTH_HC",  2))     # hc_mult
GS  = 64
KDA_PROJ = KH * KD

rng = np.random.default_rng(53)

def save_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for b in blobs: f.write(b)

def f16(o, i, s=None):
    if s is None: s = (2.0 / (o + i)) ** 0.5 * 0.5
    return (rng.standard_normal((o, i)) * s).astype(np.float16)

def f16v(n):
    return (np.ones(n, np.float32) * 0.6 + rng.random(n).astype(np.float32) * 0.3).astype(np.float16)

def f32v(n):
    return (rng.standard_normal(n) * 0.1).astype(np.float32)

def i4(o, i):
    w = (rng.standard_normal((o, i)) * (2.0 / (o + i)) ** 0.5).astype(np.float32)
    ng = i // GS
    g = w.reshape(o, ng, GS)
    amax = np.abs(g).max(axis=2, keepdims=True)
    s = np.maximum(amax / 7.0, 1e-8).astype(np.float32).reshape(o, ng)
    q = np.rint(np.clip(g / np.maximum(amax, 1e-8) * 7.0, -7, 7)).astype(np.int8).reshape(-1)
    lo = (q[0::2] & 0xF).astype(np.uint8)
    hi = ((q[1::2] & 0xF) << 4).astype(np.uint8)
    return (lo | hi, s)

HC_MIX = (2 + HC) * HC
layer_types = ["linear_attention"] * (L - 1) + ["full_attention"]
mlp_types = ["dense" if i < FD else "sparse" for i in range(L)]

cfg = {
    "architectures": ["Glm53ForCausalLM"],
    "model_type": "glm53",
    "hidden_size": H, "num_hidden_layers": L, "vocab_size": V,
    "intermediate_size": DI,
    "num_attention_heads": NH,
    "q_lora_rank": QL, "kv_lora_rank": KL,
    "qk_nope_head_dim": QK, "qk_rope_head_dim": 0, "v_head_dim": VH,
    "n_routed_experts": E, "num_experts_per_tok": TOPK,
    "moe_intermediate_size": MI, "n_shared_experts": 1,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-6, "swiglu_limit": 7.0,
    "num_nextn_predict_layers": 0,
    "hc_mult": HC, "hc_sinkhorn_iters": 2, "hc_eps": 1e-6,
    "first_k_dense_replace": FD,
    "sparse": E > 0,
    "mlp_layer_types": mlp_types,
    "layer_types": layer_types,
    "linear_attn_config": {
        "num_heads": KH, "head_dim": KD, "short_conv_kernel_size": CK,
        "gate_lower_bound": -5.0,
    },
    "index_topk": IT, "index_n_heads": IH, "index_head_dim": ID,
    "index_kpool": IP, "index_kpool_always_select_tail": True,
    "max_position_embeddings": 8192, "eos_token_id": V - 1,
}
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)

t = {}
t["model.embed_tokens.weight"] = ((rng.standard_normal((V, H)) * 0.02).astype(np.float16), "F16")
t["model.norm.weight"] = (f16v(H), "F16")

for li in range(L):
    p = f"model.layers.{li}."
    t[p + "input_layernorm.weight"] = (f16v(H), "F16")
    t[p + "post_attention_layernorm.weight"] = (f16v(H), "F16")
    for site in ("attn", "ffn"):
        t[p + f"hc_{site}_fn"] = ((rng.standard_normal(HC_MIX * HC * H) * 0.05).astype(np.float16), "F16")
        t[p + f"hc_{site}_base"] = ((rng.standard_normal(HC_MIX) * 0.1).astype(np.float16), "F16")
        t[p + f"hc_{site}_scale"] = (np.ones(3, np.float32), "F32")
    if layer_types[li] == "full_attention":
        t[p + "self_attn.q_a_proj.weight"] = (f16(QL, H), "F16")
        t[p + "self_attn.q_a_layernorm.weight"] = (f16v(QL), "F16")
        t[p + "self_attn.q_b_proj.weight"] = (f16(NH * QK, QL), "F16")
        t[p + "self_attn.kv_a_proj_with_mqa.weight"] = (f16(KL, H), "F16")   # qk_rope=0 (NoPE)
        t[p + "self_attn.kv_a_layernorm.weight"] = (f16v(KL), "F16")
        t[p + "self_attn.kv_b_proj.weight"] = (f16(NH * (QK + VH), KL), "F16")
        t[p + "self_attn.o_proj.weight"] = (f16(H, NH * VH), "F16")
        t[p + "self_attn.indexer.wq_b.weight"] = (f16(IH * ID, QL), "F16")
        t[p + "self_attn.indexer.wk.weight"] = (f16(ID, H), "F16")
        t[p + "self_attn.indexer.weights_proj.weight"] = (f16(IH, H), "F16")
        t[p + "self_attn.indexer.k_norm.weight"] = (f16v(ID), "F16")
        t[p + "self_attn.indexer.k_norm.bias"] = (f32v(ID), "F32")
        if IP > 1:
            t[p + "self_attn.indexer.index_kpool_compress_ape"] = ((rng.standard_normal(IP * ID) * 0.1).astype(np.float32), "F32")
            t[p + "self_attn.indexer.index_kpool_compress_gate"] = (f16(ID, H), "F16")
    else:
        for nm, O, I in (("q_proj", KDA_PROJ, H), ("k_proj", KDA_PROJ, H), ("v_proj", KDA_PROJ, H),
                         ("o_proj", H, KDA_PROJ), ("g_a_proj", KD, H), ("g_b_proj", KDA_PROJ, KD),
                         ("f_a_proj", KD, H), ("f_b_proj", KDA_PROJ, KD), ("b_proj", KH, H)):
            t[p + f"self_attn.{nm}.weight"] = (f16(O, I), "F16")
        for cv in ("q_conv1d", "k_conv1d", "v_conv1d"):
            t[p + f"self_attn.{cv}.weight"] = ((rng.standard_normal(KDA_PROJ * CK) * 0.1).astype(np.float32), "F32")
        t[p + "self_attn.dt_bias"] = (f32v(KDA_PROJ), "F32")
        t[p + "self_attn.A_log"] = ((-rng.random(KH) * 2).astype(np.float32), "F32")
        t[p + "self_attn.o_norm.weight"] = (f16v(KD), "F16")
    if li < FD:
        t[p + "mlp.gate_proj.weight"] = (f16(DI, H), "F16")
        t[p + "mlp.up_proj.weight"] = (f16(DI, H), "F16")
        t[p + "mlp.down_proj.weight"] = (f16(H, DI), "F16")
    else:
        t[p + "mlp.gate.weight"] = (f16(E, H), "F16")
        t[p + "mlp.gate.e_score_correction_bias"] = (f32v(E), "F32")
        for e in range(E):
            for nm, O, I in (("gate_proj", MI, H), ("up_proj", MI, H), ("down_proj", H, MI)):
                q, sc = i4(O, I)
                t[p + f"mlp.experts.{e}.{nm}.weight"] = (q, "U8")
                t[p + f"mlp.experts.{e}.{nm}.weight.qs"] = (sc, "F32")
        for nm, O, I in (("gate_proj", MI, H), ("up_proj", MI, H), ("down_proj", H, MI)):
            t[p + f"mlp.shared_experts.{nm}.weight"] = (f16(O, I), "F16")
    if li % 4 == 0:
        print(f"layer {li} done", flush=True)

save_st(f"{OUT}/model-00000.safetensors", t)
mb = os.path.getsize(f"{OUT}/model-00000.safetensors") / 1e6
print(f"container: {mb:.1f} MB, {L} layers ({layer_types.count('full_attention')} MLA + "
      f"{layer_types.count('linear_attention')} KDA), {E} experts, glm53 dialect")
