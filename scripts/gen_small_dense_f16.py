#!/usr/bin/env python3
"""gen_small_dense_f16.py — miniature DENSE F16 container in the field q27_i4 dialect
(F16 dense trunk, no routed experts, shared_inter MLP) with NON-DEGENERATE
embed/lm_head (real random rows, lm_head scaled up) so tf-nll on a correct
engine is measurably != ln(V). Geometry mirrors gen_student_i4.py, shrunk:
H=512 L=8 (attn every 4th) MLP=256 V=4096 QH=8 KVH=2 HD=64 DN 4/2/64/64/k4.

E17: this is the container the mapped-F16 numeric-parity gate needs and the
int4-packed synth family cannot provide — every dense matrix here goes through
the TRUNK_MAP F16 mmap branch (matmul_map_f16 / q36_down_prefix /
down_prefix_batch), the exact kernels the field q27_i4 F16 trunk rides, and
the exact kernels the bias-less map_f16_one silently zeroed (2^-112) before
the fix. Small enough for any CI runner or the 1 GB dev sandbox (~25 MB).
"""
import json, os, struct
import numpy as np

OUT = os.path.expanduser(os.environ.get("SMALL_OUT", "~/synth_models/small_dense_f16"))
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

H, L, MLP, V = 512, 8, 256, 4096
QH, KVH, HD = 8, 2, 64
DN_VH, DN_KH, DN_KDIM, DN_VDIM, CONVK = 4, 2, 64, 64, 4
CONV_DIM = 2*DN_KH*DN_KDIM + DN_VH*DN_VDIM

rng = np.random.default_rng(21)

def save_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off+len(b)]}
        blobs.append(b); off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for b in blobs: f.write(b)

def norm1(n):
    return ((np.ones(n, np.float32)*0.6 + rng.random(n)*0.3).astype(np.float16))

def mat_f16(o, i, s=None):
    if s is None: s = (2.0/(o+i))**0.5 * 0.5
    return (rng.standard_normal((o, i)) * s).astype(np.float16)

g = {
  # REAL random heads (the field container has trained ones; the default synth
  # container's constant 0.001 rows force logits identical across the vocab =
  # NLL exactly ln(V) BY DESIGN — blind to every tf-nll bug there is).
  "model.embed_tokens.weight": ((rng.standard_normal((V, H)) * 0.02).astype(np.float16), "F16"),
  # lm_head scaled 10x the init default: sigma_logit ~ 3.5 so a correct engine's
  # NLL lands far from ln(V) and a zero-logit engine lands exactly on it.
  "lm_head.weight":            ((rng.standard_normal((V, H)) * (10.0*(2.0/(V+H))**0.5)).astype(np.float16), "F16"),
  "model.norm.weight":         (norm1(H), "F16"),
}
save_st(f"{OUT}/model-globals.safetensors", g)

for li in range(L):
    p = f"model.layers.{li}."
    is_attn = (li % 4 == 3)
    t = {
      p+"input_layernorm.weight": (norm1(H), "F16"),
      p+"post_attention_layernorm.weight": (norm1(H), "F16"),
    }
    if is_attn:
        for nm, O, I in [("q_proj", QH*HD*2, H), ("k_proj", KVH*HD, H),
                         ("v_proj", KVH*HD, H), ("o_proj", H, QH*HD)]:
            t[p+"self_attn."+nm+".weight"] = (mat_f16(O, I), "F16")
    else:
        for nm, O, I in [("linear_attn.in_proj_qkv.weight", CONV_DIM, H),
                         ("linear_attn.in_proj_z.weight", DN_VH*DN_VDIM, H),
                         ("linear_attn.out_proj.weight", H, DN_VH*DN_VDIM)]:
            t[p+nm] = (mat_f16(O, I), "F16")
        t[p+"linear_attn.in_proj_b.weight"]   = ((rng.standard_normal(DN_VH*H)*0.05).astype(np.float16), "F16")
        t[p+"linear_attn.in_proj_a.weight"]   = ((rng.standard_normal(DN_VH*H)*0.05).astype(np.float16), "F16")
        t[p+"linear_attn.conv1d.weight"]      = ((rng.standard_normal(CONV_DIM*CONVK)*0.1).astype(np.float16).reshape(CONV_DIM, CONVK), "F16")
        t[p+"linear_attn.dt_bias"]            = ((rng.standard_normal(DN_VH)*0.05).astype(np.float16), "F16")
        t[p+"linear_attn.A_log"]              = ((rng.standard_normal(DN_VH)*0.1).astype(np.float16), "F16")
        t[p+"linear_attn.norm.weight"]        = (np.ones(DN_VDIM, dtype=np.float16), "F16")
    for nm, O, I in [("mlp.gate_proj", MLP, H), ("mlp.up_proj", MLP, H), ("mlp.down_proj", H, MLP)]:
        t[p+nm+".weight"] = (mat_f16(O, I), "F16")
    save_st(f"{OUT}/model-{li:05d}.safetensors", t)

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
  "rope_theta": 1000000.0, "rms_eps": 1e-6, "ebits": 4,
  "zero_centered_norms": True,
  "dn_vheads": DN_VH, "dn_kheads": DN_KH, "dn_kdim": DN_KDIM, "dn_vdim": DN_VDIM,
  "dn_convk": CONVK, "dn_conv_dim": CONV_DIM,
}
json.dump(meta, open(f"{OUT}/qwen36_meta.json", "w"), indent=1)

mb = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))/1e6
print(f"container: {mb:.1f} MB F16 dense, {L} layers, V={V} H={H}")
