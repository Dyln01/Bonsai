#!/usr/bin/env python3
"""gen_mtp_head_synth.py — add a synthetic MTP head to an existing qwen36 container.

Emits `model-mtp.safetensors` beside the shards and sets `"mtp": 1,
"mtp_layers": 1` in qwen36_meta.json. The head contract mirrors the real
Qwen3.6-35B-A3B checkpoint (see docs/MTP_DESIGN_QWEN36.md):

  mtp.fc.weight                        [H, 2H]      concat projection
  mtp.pre_fc_norm_embedding.weight     [H]
  mtp.pre_fc_norm_hidden.weight        [H]
  mtp.norm.weight                      [H]          final norm (pre lm_head)
  mtp.layers.0.input_layernorm.weight  [H]
  mtp.layers.0.post_attention_layernorm.weight [H]
  mtp.layers.0.self_attn.{q,k,v,o}_proj.weight    full attention only
  mtp.layers.0.self_attn.{q,k}_norm.weight        iff the container has qk norms
  mtp.layers.0.mlp.gate.weight + mlp.experts.<e>.merged_weight/qs   (MoE container)
  mtp.layers.0.mlp.shared_expert.* + mlp.shared_expert_gate          (iff shared_inter>0)
  -- or, on a dense container (num_experts==0) --
  mtp.layers.0.mlp.{gate,up,down}_proj.weight                       (the mlp. prefix)

Random weights (fixed seed): acceptance rates are meaningless by design —
the head exists to exercise the engine's loader, draft source, verify and
rollback machinery (Q36_MTP_FORCE gates).

STOCK-SAFE: an engine without MTP support never requests an mtp.* tensor and
ignores the new meta keys, so the file is inert for existing runs (verified
by the regression arm of the MTP battery).

Usage: python3 gen_mtp_head_synth.py <container_dir> [--seed 77]
"""
import json, os, struct, sys
import numpy as np

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

def main():
    args = [a for a in sys.argv[1:]]
    seed = 77
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    cont = os.path.expanduser(args[0])
    meta_p = os.path.join(cont, "qwen36_meta.json")
    meta = json.load(open(meta_p))
    if meta.get("mtp"):
        sys.exit("container already carries an MTP head (meta.mtp=1); refusing")

    H = meta["hidden"]; QH = meta["q_heads"]; KVH = meta["kv_heads"]; HD = meta["head_dim"]
    # q/k/v/o SHAPES follow the container's own meta, not naive head_dim: gated
    # containers carry q_head_dim = 2*head_dim (query ++ gate) and the loader
    # reads q_proj as [q_heads*q_head_dim, hidden], o_proj as [hidden, o_in].
    QHD = meta.get("q_head_dim", HD); KHD = meta.get("k_head_dim", HD); OIN = meta.get("o_in", QH * HD)
    E = meta.get("num_experts", 0); I = meta.get("moe_inter", 0); SI = meta.get("shared_inter", 0)
    GS = meta.get("expert_gs", 64)
    qk_norm = bool(meta.get("has_qk_norm", False))
    cfg = json.load(open(os.path.join(cont, "config.json")))
    V = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
    if not V:
        sys.exit("vocab_size not found in config.json")

    rng = np.random.default_rng(seed)
    def f16(o, i, s=None):
        if s is None: s = (2.0 / (o + i)) ** 0.5 * 0.5
        return (rng.standard_normal((o, i)) * s).astype(np.float16)
    def norm1(n):
        return (np.ones(n, np.float32) * 0.6 + rng.random(n) * 0.3).astype(np.float16)

    # int4 gs-merged expert, same packing as gen_coder30_synth.py
    def merged_expert(mlp):
        parts = []
        for (O, In) in ((mlp, H), (mlp, H), (H, mlp)):
            w = (rng.standard_normal((O, In)) * (2.0 / (O + In)) ** 0.5).astype(np.float32)
            ng = In // GS
            amax = np.abs(w.reshape(O, ng, GS)).max(axis=2, keepdims=True)
            s = np.maximum(amax / 7.0, 1e-8).astype(np.float32).reshape(O, ng)
            q = np.rint(np.clip(w.reshape(O, ng, GS) / np.maximum(amax, 1e-8) * 7.0, -7, 7)).astype(np.int8)
            parts.append((q.reshape(-1), s.reshape(-1)))
        flat_q = np.concatenate([p[0] for p in parts])
        flat_s = np.concatenate([p[1] for p in parts]).astype(np.float32)
        lo = (flat_q[0::2] & 0xF).astype(np.uint8)
        hi = ((flat_q[1::2] & 0xF) << 4).astype(np.uint8)
        return (lo | hi), flat_s

    t = {
        "mtp.fc.weight":                     (f16(H, 2 * H), "F16"),
        "mtp.pre_fc_norm_embedding.weight":  (norm1(H), "F16"),
        "mtp.pre_fc_norm_hidden.weight":     (norm1(H), "F16"),
        "mtp.norm.weight":                   (norm1(H), "F16"),
    }
    p = "mtp.layers.0."
    t[p + "input_layernorm.weight"] = (norm1(H), "F16")
    t[p + "post_attention_layernorm.weight"] = (norm1(H), "F16")
    t[p + "self_attn.q_proj.weight"] = (f16(QH * QHD, H), "F16")
    t[p + "self_attn.k_proj.weight"] = (f16(KVH * KHD, H), "F16")
    t[p + "self_attn.v_proj.weight"] = (f16(KVH * KHD, H), "F16")
    t[p + "self_attn.o_proj.weight"] = (f16(H, OIN), "F16")
    if qk_norm:
        t[p + "self_attn.q_norm.weight"] = (norm1(HD), "F16")
        t[p + "self_attn.k_norm.weight"] = (norm1(KHD), "F16")

    if E:      # MoE container: router + fused experts (+ shared expert if any)
        t[p + "mlp.gate.weight"] = (f16(E, H), "F16")
        for e in range(E):
            mw, qs = merged_expert(I)
            t[p + f"mlp.experts.{e}.merged_weight"] = (mw, "U8")
            t[p + f"mlp.experts.{e}.qs"] = (qs, "F32")
        if SI:
            t[p + "mlp.shared_expert.gate_proj.weight"] = (f16(SI, H), "F16")
            t[p + "mlp.shared_expert.up_proj.weight"]   = (f16(SI, H), "F16")
            t[p + "mlp.shared_expert.down_proj.weight"] = (f16(H, SI), "F16")
            t[p + "mlp.shared_expert_gate.weight"]      = (f16(1, H), "F16")
    else:      # dense container: the MLP rides the shared-expert path (mlp. prefix)
        t[p + "mlp.gate_proj.weight"] = (f16(SI, H), "F16")
        t[p + "mlp.up_proj.weight"]   = (f16(SI, H), "F16")
        t[p + "mlp.down_proj.weight"] = (f16(H, SI), "F16")

    save_st(os.path.join(cont, "model-mtp.safetensors"), t)
    meta["mtp"] = 1
    meta["mtp_layers"] = 1
    json.dump(meta, open(meta_p, "w"), indent=1)
    mb = os.path.getsize(os.path.join(cont, "model-mtp.safetensors")) / 1e6
    print(f"MTP head written: {len(t)} tensors, {mb:.1f} MB, "
          f"{'MoE' if E else 'dense'} head (H={H}, QH={QH}, KVH={KVH}, HD={HD}, V={V})")
    print("meta.mtp=1 set — an MTP-unaware engine ignores both (stock-safe)")

if __name__ == "__main__":
    main()
