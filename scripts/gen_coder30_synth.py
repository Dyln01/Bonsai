#!/usr/bin/env python3
"""gen_coder30_synth.py — full-shape synthetic Qwen3-Coder-30B-A3B container.

Exact qwen3_moe geometry the engine expects (docs/qwen36.md):
  48 layers, ALL full_attention (no DeltaNet), no attn output gate, no shared
  expert, 128 experts top-8 norm_topk_prob, hidden 2048, q_heads 16, kv_heads 4
  (GQA 4:1... engine derives from meta), head_dim 128 with full rotary
  (partial_rotary_factor 1.0, theta 1e7), vocab 151936? -> coder uses 151936;
  we write 151936. rope over whole head.

Format (verified against convert_qwen36.py + qwen36.c loader):
  - per-layer shard: router f16 + q/k/v/o f16 + norms f16 + 128 experts as
    merged_weight (TRUE int4 packed: low nibble = even element, signed [-8,7])
    + qs f32 (per-row scales, [gate|up|down] concatenated, gs=64 groups).
  - globals shard: embed/lm_head/model.norm f16.
  - meta: layer_types all full_attention, shared_inter 0, partial_rotary 1.0,
    zero_centered_norms false, topk 8, norm_topk_prob true, attn_output_gate false.
Random weights; Zipf router bias optional (skip — engine doesn't need it).
"""
import json, os, struct
import numpy as np

OUT = os.path.expanduser(os.environ.get("SYNTH_OUT", "~/synth_models/synth_coder30"))
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

# defaults = the 30B-class geometry; SYNTH_* overrides let CI gate a small MoE
# with the same code paths (router/groups/int4-gs64 experts) in seconds.
H   = int(os.environ.get("SYNTH_H",   2048))
L   = int(os.environ.get("SYNTH_L",   48))
E   = int(os.environ.get("SYNTH_E",   128))
K   = int(os.environ.get("SYNTH_K",   8))
MLP = int(os.environ.get("SYNTH_MLP", 768))   # moe_inter 768 -> 30.2B expert params
QH, KVH, HD, V = 16, 4, 128, 151936
GS = 64

# param check: real coder30 = 30.5B. 48 layers x 128 experts x 3 x 2048 x 1024 = 40.3B? too big.
# Real: expert inter = 1024, so 48*128*3*1024*2048 = 40.3B — that's the actual total params
# (30B counts... let's just match the docs table: hidden 2048, 128 experts, topk 8; the
#  Coder-30B config says moe_intermediate_size 1024? No — 512? We use 1024 and verify loader.)

rng = np.random.default_rng(33)

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

def f16(o, i, s=None):
    if s is None: s = (2.0/(o+i))**0.5 * 0.5
    return (rng.standard_normal((o, i))*s).astype(np.float16)

def norm1(n, zero_centered=False):
    base = np.ones(n, np.float32) if zero_centered else (np.ones(n, np.float32)*0.6 + rng.random(n)*0.3)
    return base.astype(np.float16)

# int4 gs64 merged expert: gate[I,H], up[I,H], down[H,I] -> packed u8 (low nibble even idx)
def merged_expert():
    parts = []
    for (O, I) in ((MLP, H), (MLP, H), (H, MLP)):
        w = (rng.standard_normal((O, I)) * (2.0/(O+I))**0.5).astype(np.float32)
        ng = I // GS
        amax = np.abs(w.reshape(O, ng, GS)).max(axis=2, keepdims=True)   # [O, ng, 1]
        s = np.maximum(amax/7.0, 1e-8).astype(np.float32).reshape(O, ng)  # [O, ng]
        q = np.rint(np.clip(w.reshape(O,ng,GS)/np.maximum(amax,1e-8)*7.0, -7, 7)).astype(np.int8)
        parts.append((q.reshape(-1), s.reshape(-1)))
    flat_q = np.concatenate([p[0] for p in parts])          # signed 4-bit [-7,7]
    flat_s = np.concatenate([p[1] for p in parts]).astype(np.float32)
    lo = (flat_q[0::2] & 0xF).astype(np.uint8)
    hi = ((flat_q[1::2] & 0xF) << 4).astype(np.uint8)
    packed = (lo | hi)
    return packed, flat_s

# ---- globals ----
g = {
  "model.embed_tokens.weight": (f16(V, H, 0.02), "F16"),
  "lm_head.weight":            (f16(V, H, 0.02), "F16"),
  "model.norm.weight":         (norm1(H, zero_centered=False), "F16"),
}
save_st(f"{OUT}/model-globals.safetensors", g)
print("globals done")

# ---- per-layer shards ----
for li in range(L):
    p = f"model.layers.{li}."
    t = {
      p+"input_layernorm.weight": (norm1(H, zero_centered=False), "F16"),
      p+"post_attention_layernorm.weight": (norm1(H, zero_centered=False), "F16"),
      p+"self_attn.q_proj.weight": (f16(QH*HD, H), "F16"),
      p+"self_attn.k_proj.weight": (f16(KVH*HD, H), "F16"),
      p+"self_attn.v_proj.weight": (f16(KVH*HD, H), "F16"),
      p+"self_attn.o_proj.weight": (f16(H, QH*HD), "F16"),
      p+"mlp.gate.weight": (f16(E, H), "F16"),   # router
    }
    for e in range(E):
        mw, qs = merged_expert()
        t[p+f"mlp.experts.{e}.merged_weight"] = (mw, "U8")
        t[p+f"mlp.experts.{e}.qs"] = (qs, "F32")
    save_st(f"{OUT}/model-{li:05d}.safetensors", t)
    if li % 8 == 0: print(f"layer {li} done", flush=True)

# ---- config ----
cfg = {
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "hidden_size": H, "num_hidden_layers": L,
  "num_attention_heads": QH, "num_key_value_heads": KVH, "head_dim": HD,
  "intermediate_size": MLP, "moe_intermediate_size": MLP,
  "num_experts": E, "num_experts_per_tok": K,
  "norm_topk_prob": True, "decoder_sparse_step": 1,
  "vocab_size": V, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
  "max_position_embeddings": 262144, "eos_token_id": [V-1],
}
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)

meta = {
  "model_type": "qwen3_moe", "hidden": H, "n_layers": L,
  "n_active": L,
  "layer_types": ["full_attention"]*L,
  "q_heads": QH, "kv_heads": KVH, "head_dim": HD,
  "q_head_dim": HD, "k_head_dim": HD, "v_head_dim": HD, "o_in": QH*HD,
  "num_experts": E, "topk": K, "moe_inter": MLP, "shared_inter": 0,
  "n_group": 1, "topk_group": 1, "attn_output_gate": False, "norm_topk_prob": True,
  "has_bias": False, "has_qk_norm": False, "partial_rotary_factor": 1.0,
  "rope_theta": 10000000.0, "rms_eps": 1e-6, "ebits": 4, "expert_gs": GS,
  "zero_centered_norms": False,
}
json.dump(meta, open(f"{OUT}/qwen36_meta.json", "w"), indent=1)

mb = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))/1e6
print(f"container: {mb/1000:.1f} GB, {L} layers x {E} experts, int4 gs{GS}")
