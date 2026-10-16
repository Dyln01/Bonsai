#!/usr/bin/env python3
"""gen_qwen38_27b_f16.py — FULL-SIZE Qwen3.8-27B-shaped dense container, F16 (converter format).

Real geometry (docs/qwen36.md): 64 layers, hidden 5120, 16 GQA attn (i%4==3) + 48 DeltaNet,
MLP 17408, q_heads 40 (kv_heads 8, head_dim 128), vocab 248320, rope_theta 1e6.
All tensors F16 like convert_qwen36.py writes. ~25 GB on disk. Random weights (no download);
TRUNK_MAP correctness is format-level (mmap + f16 dequant), which random weights fully test.
"""
import json, os, struct
import numpy as np

OUT = os.path.expanduser("~/synth_models/synth_qwen38_27b_f16")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

H, L, MLP, V = 5120, 64, 17408, 248320
QH, KVH, HD = 40, 8, 128
DN_VH, DN_KH, DN_KDIM, DN_VDIM, CONVK = 32, 8, 128, 128, 4
CONV_DIM = 2*DN_KH*DN_KDIM + DN_VH*DN_VDIM     # 4096

rng = np.random.default_rng(27)

def save_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():          # name -> np array (already f16/f32)
        b = arr.tobytes()
        dt = "F16" if arr.dtype == np.float16 else "F32"
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off+len(b)]}
        blobs.append(b); off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for b in blobs: f.write(b)

def f16mat(o, i, s=None):
    if s is None: s = (2.0/(o+i))**0.5 * 0.5
    return (rng.standard_normal((o, i))*s).astype(np.float16)

def norm1(n):
    return ((np.ones(n, np.float32)*0.6 + rng.random(n)*0.3).astype(np.float16))

# globals (embed+head stay F16; the engine gathers embed rows and reads lm_head
# through matmul — under TRUNK_MAP lm_head is mmap'd too when it comes from load_tq.
# NOTE: in qwen36.c embed/lm_head load via load_t_n/embed path, not load_tq, so they
# stay RESIDENT f32->int8 — 2 x 248320x5120 f32 staging briefly. That's the engine's
# own handling; TRUNK_MAP leaves it as-is for this test.)
g = {
  "model.embed_tokens.weight": f16mat(V, H, 0.02),
  "lm_head.weight":            f16mat(V, H, 0.02),
  "model.norm.weight":         norm1(H),
}
save_st(f"{OUT}/model-globals.safetensors", g)
print("globals done", flush=True)

for li in range(L):
    p = f"model.layers.{li}."
    is_attn = (li % 4 == 3)
    t = {
      p+"input_layernorm.weight": norm1(H),
      p+"post_attention_layernorm.weight": norm1(H),
    }
    if is_attn:
        # qwen3_5: q_proj outputs query+gate per head -> 2*QH*HD
        for nm, O, I in [("q_proj", QH*HD*2, H), ("k_proj", KVH*HD, H),
                         ("v_proj", KVH*HD, H), ("o_proj", H, QH*HD)]:
            t[p+"self_attn."+nm+".weight"] = f16mat(O, I)
    else:
        t[p+"linear_attn.in_proj_qkv.weight"] = f16mat(CONV_DIM, H)
        t[p+"linear_attn.in_proj_z.weight"]   = f16mat(DN_VH*DN_VDIM, H)
        t[p+"linear_attn.out_proj.weight"]    = f16mat(H, DN_VH*DN_VDIM)
        t[p+"linear_attn.in_proj_b.weight"]   = ((rng.standard_normal(DN_VH*H)*0.05).astype(np.float16))
        t[p+"linear_attn.in_proj_a.weight"]   = ((rng.standard_normal(DN_VH*H)*0.05).astype(np.float16))
        t[p+"linear_attn.conv1d.weight"]      = ((rng.standard_normal(CONV_DIM*CONVK)*0.1).astype(np.float16).reshape(CONV_DIM, CONVK))
        t[p+"linear_attn.dt_bias"]            = ((rng.standard_normal(DN_VH)*0.05).astype(np.float16))
        t[p+"linear_attn.A_log"]              = ((rng.standard_normal(DN_VH)*0.1).astype(np.float16))
        t[p+"linear_attn.norm.weight"]        = np.ones(DN_VDIM, dtype=np.float16)
    for nm, O, I in [("mlp.gate_proj", MLP, H), ("mlp.up_proj", MLP, H), ("mlp.down_proj", H, MLP)]:
        t[p+nm+".weight"] = f16mat(O, I)
    save_st(f"{OUT}/model-{li:05d}.safetensors", t)
    if li % 8 == 0: print("layer", li, "done", flush=True)

cfg = {
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "hidden_size": H, "num_hidden_layers": L,
  "num_attention_heads": QH, "num_key_value_heads": KVH, "head_dim": HD,
  "intermediate_size": MLP, "num_experts": 0, "num_experts_per_tok": 0,
  "moe_intermediate_size": MLP, "n_shared_experts": 1,
  "vocab_size": V, "rms_norm_eps": 1e-6,
  "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
  "max_position_embeddings": 262144, "eos_token_id": [V-1],
}
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)

meta = {
  "model_type": "qwen3_5", "hidden": H, "n_layers": L, "n_active": 0,
  "q_heads": QH, "kv_heads": KVH, "head_dim": HD,
  "q_head_dim": HD*2, "k_head_dim": HD, "v_head_dim": HD, "o_in": QH*HD,
  "num_experts": 0, "topk": 0, "moe_inter": 0, "shared_inter": MLP,
  "n_group": 1, "topk_group": 1, "attn_output_gate": True, "norm_topk_prob": False,
  "has_bias": False, "has_qk_norm": False, "partial_rotary_factor": 0.25,
  "rope_theta": 1000000.0, "rms_eps": 1e-6, "ebits": 8,
  "zero_centered_norms": True,
  "dn_vheads": DN_VH, "dn_kheads": DN_KH, "dn_kdim": DN_KDIM, "dn_vdim": DN_VDIM,
  "dn_convk": CONVK, "dn_conv_dim": CONV_DIM,
}
json.dump(meta, open(f"{OUT}/qwen36_meta.json", "w"), indent=1)

mb = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))/1e6
print(f"container: {mb/1000:.1f} GB F16, {L} layers, hidden {H}, vocab {V}")
