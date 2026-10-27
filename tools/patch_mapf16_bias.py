#!/usr/bin/env python3
"""patch_mapf16_bias.py — the mapped-F16 exponent-bias fix (E17).

MAPFAST v3.1 (patch_hotstack_v1.py) shipped map_f16_one WITHOUT the F16->F32
exponent-bias rebase: the raw 5-bit exponent (bias 15) was dropped straight
into the 8-bit f32 exponent field (bias 127), so every normal F16 value read
back 2^-112 (~1.9e-34) of its true value. On an F16-dense container under
TRUNK_MAP that silently removes the ENTIRE trunk from the computation:

  * every mapped GEMV output (attn q/k/v/o, DeltaNet projections, shared-MLP
    gate/up/down, lm_head) is ~1e-34 — numerically zero against the residual;
  * residuals freeze at the embedding row (DUMP_LAYERS: identical rows, zero
    sublayer outputs, all layers);
  * tf-NLL == ln(V) EXACTLY (logits ~1e-33 -> softmax uniform to the last
    float) — field Phase-5 arm A: ppl = 248320.00 = vocab_size;
  * greedy decode still emits deterministic tokens (uniform positive scaling
    preserves argmax order until underflow) — the field's 'чисwik' (E13) and
    the byte-identical outputs across EVERY battery arm (E14 'arm-invariance')
    were this, not unsorted-pruning behavior;
  * the E8 CI gate's "logits underflow float32 norms (~1e-35)" comment was
    the same signature, written off as synthetic-container weirdness.

Why every CI gate stayed green: all synthetic containers are int4-packed, so
the numeric gates ran matmul_map_i4 (int8 nibbles x f32 .qs scales read via
st_read_f32 — no map_f16_one); the resident paths use stock st.h's correct
f16_to_f32_bulk (exp + 112). Only F16-dense TRUNK_MAP runs — i.e. only the
field host — hit the broken converter. Two sites carried it: map_f16_one
(matmul_map_f16, q36_down_prefix_batch, the GRIFFIN gather) and a second
inline copy in q36_down_prefix's f16 branch; both are fixed here, the inline
copy by folding it onto the corrected helper.

Anchored exact-byte edits, house style. Backup qwen36.c.premapf16.
"""
import hashlib
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "bias 15 -> 127" in src:
    raise SystemExit("patch_mapf16_bias: already applied")

OLD_A = r"""/* MAPFAST v3.1: stage one output row's dequantized f32 per thread, then the
 * same fixed-lane dot the resident kernels use, over all S input rows. */
static inline float map_f16_one(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu) << 23;
    uint32_t man  = (uint32_t)(h & 0x3FFu) << 13;
    uint32_t bits = sign | exp | man;
    if (((h >> 10) & 0x1F) == 0) bits = sign;   /* subnormal -> +-0 */
    float f; memcpy(&f, &bits, 4); return f;
}
"""

NEW_A = r"""/* MAPFAST v3.1: stage one output row's dequantized f32 per thread, then the
 * same fixed-lane dot the resident kernels use, over all S input rows.
 * E17: v3.1 shipped this WITHOUT the F16->F32 exponent-bias rebase — the raw
 * 5-bit exponent (bias 15) went straight into the 8-bit field (bias 127), so
 * every normal F16 read back 2^-112 of its true value and the whole mapped
 * trunk was numerically zero (frozen-embed residuals, tf-NLL exactly ln V,
 * greedy decode = argmax of embed->lm_head alone). Correct scalar form of
 * stock st.h's f16_to_f32_bulk (exp + 112), inf/nan included. */
static inline float map_f16_one(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t man  = (uint32_t)(h & 0x3FFu) << 13;
    uint32_t bits;
    if (exp == 0)       bits = sign;                      /* subnormal -> +-0 */
    else if (exp == 31) bits = sign | 0x7F800000u | man;  /* inf / nan */
    else                bits = sign | ((exp + 112u) << 23) | man;  /* bias 15 -> 127 */
    float f; memcpy(&f, &bits, 4); return f;
}
"""

OLD_B = r"""            for (int i = 0; i < Ik; i++) {
                uint16_t h = row[i];
                uint32_t bits = ((uint32_t)(h & 0x8000u) << 16) | ((uint32_t)((h >> 10) & 0x1Fu) << 23) | ((uint32_t)(h & 0x3FFu) << 13);
                if (((h >> 10) & 0x1F) == 0) bits = (uint32_t)(h & 0x8000u) << 16;
                float wv; memcpy(&wv, &bits, 4);
                acc += wv * g[i];
            }
"""

NEW_B = r"""            for (int i = 0; i < Ik; i++)
                acc += map_f16_one(row[i]) * g[i];   /* E17: was a second inline copy of the bias-less conversion */
"""

for tag, old in (("A(map_f16_one)", OLD_A), ("B(down_prefix inline)", OLD_B)):
    if src.count(old) != 1:
        raise SystemExit("patch_mapf16_bias: anchor %s count=%d (want 1)" % (tag, src.count(old)))
src = src.replace(OLD_A, NEW_A, 1).replace(OLD_B, NEW_B, 1)

bak = P + ".premapf16"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mapf16_bias applied (2 edits, backup %s)" % bak)
print("  in  md5 %s" % hashlib.md5(open(bak, "rb").read()).hexdigest())
print("  out md5 %s" % hashlib.md5(open(P, "rb").read()).hexdigest())
