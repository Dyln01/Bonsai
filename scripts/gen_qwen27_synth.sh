#!/bin/bash
# Generate a Qwen3.8-27B-shaped SYNTHETIC dense model for the qwen36 engine
# (scaled-down hidden dim so it fits; same architecture: Gated DeltaNet, no MoE).
set -e
# make_synthetic_model.py is reconstructed in-tree (scripts/); Q36_SYNTHGEN can
# still point at the original out-of-repo module dir if you have it.
if [ -n "${Q36_SYNTHGEN:-}" ]; then
  cd "$Q36_SYNTHGEN"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

# qwen36 dense: num_experts=0, MLP -> shared expert, GQA attention.
# Real 27B: hidden 5120, 64 layers (16 attn + 48 delta), MLP 17408, kv_heads 8, head_dim 128.
# Synthetic (small, still stresses KV + dense bandwidth): hidden 1024, 24 layers,
#   kv_heads 4, head_dim 64, MLP 2048. All dense; f32 norms/router-free.
python3 - <<'PYEOF'
import json, os, sys
sys.path.insert(0, ".")
from make_synthetic_model import save_safetensors, quant_int4_grouped, quant_int8_row
import numpy as np

rng = np.random.default_rng(7)
OUT = os.environ.get("Q36_HOME", "/home/dylan") + "/synth_models/synth_qwen27"
os.makedirs(OUT, exist_ok=True)
H, L, KVH, HD, MLP, V = 1024, 24, 4, 64, 2048, 4096
QH = 8

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
# qwen36_meta.json for the dense variant (per docs/qwen36.md the converter writes it)
json.dump({"num_experts": 0, "dense_intermediate": MLP}, open(f"{OUT}/qwen36_meta.json", "w"))

def rmat(o, i, s=None):
    if s is None: s = (2.0/(o+i))**0.5 * 0.5
    return (rng.standard_normal((o, i)) * s).astype(np.float32)

t = {}
# embed + lm_head (int8 row)
emb = (rng.standard_normal((V, H)) * 0.02).astype(np.float32)
q8, s8 = quant_int8_row(emb)
t["model.embed_tokens.weight"] = (q8, "I8")
t["model.embed_tokens.weight.qs"] = (s8, "F32")
head = (rng.standard_normal((V, H)) * 0.02).astype(np.float32)
qh, sh = quant_int8_row(head)
t["lm_head.weight"] = (qh, "I8")
t["lm_head.weight.qs"] = (sh, "F32")
t["model.norm.weight"] = (np.ones(H, np.float32)*0.6 + rng.random(H)*0.3, "F32")

for li in range(L):
    p = f"model.layers.{li}."
    for nm in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
        t[p+nm] = (np.ones(H, np.float32)*0.6 + rng.random(H)*0.3, "F32")
    # qwen36 attn: q_proj outputs (query+gate), k/v head_dim each
    for nm, O, I in [("q_proj", QH*HD*2, H), ("k_proj", KVH*HD, H),
                     ("v_proj", KVH*HD, H), ("o_proj", H, QH*HD)]:
        w = rmat(O, I)
        q, s = quant_int4_grouped(w)
        t[p+"self_attn."+nm+".weight"] = (q, "U8")
        t[p+"self_attn."+nm+".weight.qs"] = (s, "F32")
    # dense MLP as shared expert: gate/up/down
    for nm, O, I in [("mlp.gate_proj", MLP, H), ("mlp.up_proj", MLP, H), ("mlp.down_proj", H, MLP)]:
        w = rmat(O, I)
        q, s = quant_int4_grouped(w)
        t[p+nm+".weight"] = (q, "U8")
        t[p+nm+".weight.qs"] = (s, "F32")

save_safetensors(f"{OUT}/model-00000.safetensors", t)
mb = sum(a.nbytes for a,_ in t.values())/1e6
print(f"wrote {OUT}: {mb:.1f} MB, {L} layers, hidden {H}")
PYEOF
python3 "$(dirname "$0")/make_tokenizer.py" "${Q36_HOME:-/home/dylan}/synth_models/synth_qwen27/tokenizer.json" 4095
du -sh "${Q36_HOME:-/home/dylan}/synth_models/synth_qwen27"
echo QWEN_SYNTH_DONE
