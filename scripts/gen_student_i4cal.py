#!/usr/bin/env python3
"""gen_student_i4cal.py — CALIBRATED hotness-sorted int4 container.

Same seeded weight draws as gen_student_i4s.py, but the neuron order comes from the
MEASURED engine profile (hot_profile.json): neurons sorted by observed |silu(g)*u| energy,
descending. For a real checkpoint, the same permutation applies to the converted container.
"""
import json, os, struct, sys
import numpy as np

OUT = os.path.expanduser("~/synth_models/student_1b_i4cal")
PROFILE = os.path.expanduser("~/diag/hot_profile.json")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

H, L, MLP, V = 1280, 20, 3456, 248320
QH, KVH, HD = 16, 4, 64
DN_VH, DN_KH, DN_KDIM, DN_VDIM, CONVK = 4, 2, 64, 64, 4
CONV_DIM = 2*DN_KH*DN_KDIM + DN_VH*DN_VDIM
rng = np.random.default_rng(21)

# ---- load the measured profile: per-layer descending permutation ----
prof = json.load(open(PROFILE))
layer_order = {}
for key, e in prof.items():
    li = int(key[1:])
    v = np.array(e["energy"], dtype=np.float64)
    order = np.argsort(-v, kind="stable")      # descending energy
    layer_order[li] = order
print(f"profile loaded: {len(layer_order)} layers")

def save_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off+len(b)]}
        blobs.append(b); off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for b in blobs: f.write(b)

def norm1(n):
    return ((np.ones(n, np.float32)*0.6 + rng.random(n)*0.3).astype(np.float16))

def quant_i4_row(w):
    w = w.astype(np.float32); O, I = w.shape
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax/7.0, 1e-8).astype(np.float32).reshape(-1)
    q = np.rint(np.clip(w/np.maximum(amax,1e-8)*7.0, -7, 7)).astype(np.int8).reshape(-1)
    lo = (q[0::2] & 0xF).astype(np.uint8); hi = ((q[1::2] & 0xF) << 4).astype(np.uint8)
    return (lo | hi), s

def mat_i4(o, i, s=None):
    if s is None: s = (2.0/(o+i))**0.5 * 0.5
    w = (rng.standard_normal((o, i)) * s)
    return quant_i4_row(w)

g = {
  "model.embed_tokens.weight": ((rng.standard_normal((V, H))*0.02).astype(np.float16), "F16"),
  "lm_head.weight":            ((rng.standard_normal((V, H))*0.02).astype(np.float16), "F16"),
  "model.norm.weight":         (norm1(H), "F16"),
}
save_st(f"{OUT}/model-globals.safetensors", g)

for li in range(L):
    p = f"model.layers.{li}."
    is_attn = (li % 4 == 3)
    t = { p+"input_layernorm.weight": (norm1(H), "F16"),
          p+"post_attention_layernorm.weight": (norm1(H), "F16") }
    if is_attn:
        for nm, O, I in [("q_proj", QH*HD*2, H), ("k_proj", KVH*HD, H),
                         ("v_proj", KVH*HD, H), ("o_proj", H, QH*HD)]:
            w, s = mat_i4(O, I)
            t[p+"self_attn."+nm+".weight"] = (w, "U8"); t[p+"self_attn."+nm+".weight.qs"] = (s, "F32")
    else:
        for nm, O, I in [("linear_attn.in_proj_qkv.weight", CONV_DIM, H),
                         ("linear_attn.in_proj_z.weight", DN_VH*DN_VDIM, H),
                         ("linear_attn.out_proj.weight", H, DN_VH*DN_VDIM)]:
            w, s = mat_i4(O, I)
            t[p+nm] = (w, "U8"); t[p+nm+".qs"] = (s, "F32")
        t[p+"linear_attn.in_proj_b.weight"]   = ((rng.standard_normal(DN_VH*H)*0.05).astype(np.float16), "F16")
        t[p+"linear_attn.in_proj_a.weight"]   = ((rng.standard_normal(DN_VH*H)*0.05).astype(np.float16), "F16")
        t[p+"linear_attn.conv1d.weight"]      = ((rng.standard_normal(CONV_DIM*CONVK)*0.1).astype(np.float16).reshape(CONV_DIM, CONVK), "F16")
        t[p+"linear_attn.dt_bias"]            = ((rng.standard_normal(DN_VH)*0.05).astype(np.float16), "F16")
        t[p+"linear_attn.A_log"]              = ((rng.standard_normal(DN_VH)*0.1).astype(np.float16), "F16")
        t[p+"linear_attn.norm.weight"]        = (np.ones(DN_VDIM, dtype=np.float16), "F16")
    # MLP: draw base weights, then permute neurons by the calibrated order
    perm = layer_order.get(li, np.arange(MLP))
    wg = (rng.standard_normal((MLP, H)) * (2.0/(MLP+H))**0.5 * 0.5)
    wu = (rng.standard_normal((MLP, H)) * (2.0/(MLP+H))**0.5 * 0.5)
    wd = (rng.standard_normal((H, MLP)) * (2.0/(H+MLP))**0.5 * 0.5)
    wg = wg[perm]; wu = wu[perm]; wd = wd[:, perm]
    q, s = quant_i4_row(wg); t[p+"mlp.gate_proj.weight"] = (q, "U8"); t[p+"mlp.gate_proj.weight.qs"] = (s, "F32")
    q, s = quant_i4_row(wu); t[p+"mlp.up_proj.weight"]   = (q, "U8"); t[p+"mlp.up_proj.weight.qs"]   = (s, "F32")
    q, s = quant_i4_row(wd); t[p+"mlp.down_proj.weight"] = (q, "U8"); t[p+"mlp.down_proj.weight.qs"] = (s, "F32")
    save_st(f"{OUT}/model-{li:05d}.safetensors", t)

cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
  "hidden_size": H, "num_hidden_layers": L, "num_attention_heads": QH,
  "num_key_value_heads": KVH, "head_dim": HD, "intermediate_size": MLP,
  "num_experts": 0, "num_experts_per_tok": 0, "moe_intermediate_size": MLP,
  "n_shared_experts": 1, "vocab_size": V, "rms_norm_eps": 1e-6,
  "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
  "max_position_embeddings": 262144, "eos_token_id": [V-1]}
json.dump(cfg, open(f"{OUT}/config.json","w"), indent=1)
meta = {"model_type":"qwen3_5","hidden":H,"n_layers":L,"n_active":0,"q_heads":QH,"kv_heads":KVH,
  "head_dim":HD,"q_head_dim":HD*2,"k_head_dim":HD,"v_head_dim":HD,"o_in":QH*HD,
  "num_experts":0,"topk":0,"moe_inter":0,"shared_inter":MLP,"n_group":1,"topk_group":1,
  "attn_output_gate":True,"norm_topk_prob":False,"has_bias":False,"has_qk_norm":False,
  "partial_rotary_factor":0.25,"rope_theta":1000000.0,"rms_eps":1e-6,"ebits":4,
  "zero_centered_norms":True,"dn_vheads":DN_VH,"dn_kheads":DN_KH,"dn_kdim":DN_KDIM,
  "dn_vdim":DN_VDIM,"dn_convk":CONVK,"dn_conv_dim":CONV_DIM}
json.dump(meta, open(f"{OUT}/qwen36_meta.json","w"), indent=1)
mb = sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))/1e6
print(f"calibrated container: {mb:.0f} MB")
