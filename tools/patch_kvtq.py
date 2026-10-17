#!/usr/bin/env python3
"""patch_kvtq.py — wire KV_TQ (kv_tq.h: PolarQuant / rotated-int4 KV rows) into qwen36.c.

Fourth KV tier for the qwen36 engine, after f32 and KV8:
  f32   : 4 B/element, lossless
  KV8=1 : 1 B/element fp8-e4m3 + per-row f32 scale        (kv8_fused.h)
  KV_TQ : packed row, width-dependent —
            codec 1 (default): rotated-int4, n/2 B/row, ~0.10 rel L2
            codec 0          : PolarQuant, (n/2)*b + (n/2-1)*(b-1) bits/row
          + per-row f32 radius in the SAME Kscale/Vscale slot KV8 already uses,
          so every read path keeps KV8's scale-hoisting: decode the row at
          radius 1 (unit direction) and multiply by Kscale/Vscale outside the
          element loop. dequant(r=1)*s == dequant(s) exactly (linear in r).

Env knobs (mutually exclusive; KV_TQ wins if both set):
  KV_TQ=<bits 2..8>     level-1 bits for PolarQuant (3|4 typical; codec 1 ignores it)
  KV_TQ_CODEC=0|1       1 = rotated-int4 (default, cheap decode), 0 = PolarQuant

Every edit is an exact-byte anchored replacement (sub1 with count==1 asserts),
applied to qwen36.c in place. Back up to qwen36.c.pretq first. Refuses to run
twice (idempotence guard: 'kv_tq' already present).

Usage: python3 patch_kvtq.py <path-to-qwen36.c>
"""
import os, shutil, sys

EDITS = []

def edit(old, new):
    EDITS.append((old, new))

# ---------------------------------------------------------------- E1: include + dispatch helper
edit(
'#include "kv8_fused.h"\n'
'/* KV8 tile staging: rows dequantized per pass; 256 rows x 64 f32 = 64 KiB/side */\n'
'#define KV8_TILE_ROWS 256   /* KV8: fp8-e4m3 KV rows (opt-in, qwen36 port of the GLM engine path) */\n',
'#include "kv8_fused.h"\n'
'/* KV8 tile staging: rows dequantized per pass; 256 rows x 64 f32 = 64 KiB/side */\n'
'#define KV8_TILE_ROWS 256   /* KV8: fp8-e4m3 KV rows (opt-in, qwen36 port of the GLM engine path) */\n'
'#include "kv_tq.h"   /* KV_TQ=<bits> [KV_TQ_CODEC=0|1]: packed KV rows, GLM-side codecs (PolarQuant / rotated int4) */\n'
'\n'
'/* KV_TQ row dequant into staging WITHOUT the per-row scale, mirroring\n'
' * kv8_dequant_noscale: the radius scales the reconstruction linearly\n'
' * (y = radius * unit direction), so dequant(r=1)*scale == dequant(r=scale)\n'
' * and the caller keeps KV8\'s scale hoisting (multiply once per row, outside\n'
' * the element loop). tq==0 falls through to the KV8 decoder. */\n'
'static inline void q36_kv_dequant_noscale(const uint8_t *src, float *dst, int kvd, int tq, int bits, int codec) {\n'
'    if (tq) coli_kvq_dequant_row(src, 1.f, dst, kvd, bits, codec);\n'
'    else kv8_dequant_noscale(src, dst, kvd);\n'
'}\n'
)

# ---------------------------------------------------------------- E2: Model struct fields
edit(
'    int kv8;                     /* KV8=1: K/V stored as fp8-e4m3 + per-row f32 scale (Kscale/Vscale) */\n',
'    int kv8;                     /* KV8=1: K/V stored as fp8-e4m3 + per-row f32 scale (Kscale/Vscale) */\n'
'    int kv_tq, tq_bits, tq_codec, kvq_row_b;  /* KV_TQ=bits [codec]: packed rows (kvq_row_b B/row) + per-row f32 radius */\n'
)

# ---------------------------------------------------------------- E3: KV write path
edit(
'        if (m->kv8) {\n'
'            float *Ks = mr ? mr[s].seq->Kscale[layer] : m->Kscale[layer];\n'
'            float *Vs = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'            Ks[(int64_t)kvh*m->max_t + t] = coli_kv8_quant_row(\n',
'        if (m->kv_tq) {\n'
'            float *Ks = mr ? mr[s].seq->Kscale[layer] : m->Kscale[layer];\n'
'            float *Vs = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'            uint8_t *K8 = (uint8_t*)Kl, *V8 = (uint8_t*)Vl;\n'
'            int rb = m->kvq_row_b;\n'
'            Ks[(int64_t)kvh*m->max_t + t] = coli_kvq_quant_row(\n'
'                k + (int64_t)s*KV*kvd + kvh*kvd, K8 + ((int64_t)kvh*m->max_t + t)*rb, kvd, m->tq_bits, m->tq_codec);\n'
'            Vs[(int64_t)kvh*m->max_t + t] = coli_kvq_quant_row(\n'
'                vv + (int64_t)s*KV*kvd + kvh*kvd, V8 + ((int64_t)kvh*m->max_t + t)*rb, kvd, m->tq_bits, m->tq_codec);\n'
'        } else if (m->kv8) {\n'
'            float *Ks = mr ? mr[s].seq->Kscale[layer] : m->Kscale[layer];\n'
'            float *Vs = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'            Ks[(int64_t)kvh*m->max_t + t] = coli_kv8_quant_row(\n'
)

# ---------------------------------------------------------------- E4: flash debug line
edit(
'        if (getenv("Q36_FLASH_DEBUG")) fprintf(stderr, "[flash] engaged S=%d QB=%d TB=%d H=%d kv8=%d\\n", S, QB, TB, H, m->kv8);\n',
'        if (getenv("Q36_FLASH_DEBUG")) fprintf(stderr, "[flash] engaged S=%d QB=%d TB=%d H=%d kv8=%d tq=%d\\n", S, QB, TB, H, m->kv8, m->kv_tq);\n'
)

# ---------------------------------------------------------------- E5: flash staging alloc + OOM gate
edit(
'            float *kq  = m->kv8 ? malloc((size_t)TB*kvd*sizeof(float)) : NULL;\n'
'            float *vq  = m->kv8 ? malloc((size_t)TB*kvd*sizeof(float)) : NULL;\n'
'            if (!sc || !acc || !mxv || !lvs || (m->kv8 && (!kq || !vq))) { fprintf(stderr, "OOM flash attn\\n"); exit(1); }\n',
'            float *kq  = (m->kv8 || m->kv_tq) ? malloc((size_t)TB*kvd*sizeof(float)) : NULL;\n'
'            float *vq  = (m->kv8 || m->kv_tq) ? malloc((size_t)TB*kvd*sizeof(float)) : NULL;\n'
'            if (!sc || !acc || !mxv || !lvs || ((m->kv8 || m->kv_tq) && (!kq || !vq))) { fprintf(stderr, "OOM flash attn\\n"); exit(1); }\n'
)

# ---------------------------------------------------------------- E6: flash scale pointers + packed stride
edit(
'                    const float *Ksc = m->kv8 ? m->Kscale[layer] : NULL;\n'
'                    const float *Vsc = m->kv8 ? m->Vscale[layer] : NULL;\n'
'                    int64_t kbase = (int64_t)kvh*m->max_t;\n',
'                    const float *Ksc = (m->kv8 || m->kv_tq) ? m->Kscale[layer] : NULL;\n'
'                    const float *Vsc = (m->kv8 || m->kv_tq) ? m->Vscale[layer] : NULL;\n'
'                    const int rb = m->kv_tq ? m->kvq_row_b : kvd;   /* packed-row stride (bytes) */\n'
'                    int64_t kbase = (int64_t)kvh*m->max_t;\n'
)

# ---------------------------------------------------------------- E7: flash K dequant
edit(
'                            const float *ksrc;\n'
'                            if (m->kv8) {\n'
'                                kv8_dequant_noscale(Kl8 + (kbase + t0 + tt)*kvd, kq + (size_t)tt*kvd, kvd);\n'
'                                ksrc = kq + (size_t)tt*kvd;\n'
'                            } else ksrc = Kl_f + (kbase + t0 + tt)*kvd;\n',
'                            const float *ksrc;\n'
'                            if (m->kv8 || m->kv_tq) {\n'
'                                q36_kv_dequant_noscale(Kl8 + (kbase + t0 + tt)*rb, kq + (size_t)tt*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
'                                ksrc = kq + (size_t)tt*kvd;\n'
'                            } else ksrc = Kl_f + (kbase + t0 + tt)*kvd;\n'
)

# ---------------------------------------------------------------- E8: flash dot scale hoist
edit(
'                                          * (m->kv8 ? Ksc[kbase + t0 + tt] * scale : scale)\n',
'                                          * ((m->kv8 || m->kv_tq) ? Ksc[kbase + t0 + tt] * scale : scale)\n'
)

# ---------------------------------------------------------------- E9: flash V dequant
edit(
'                        if (m->kv8)\n'
'                            for (int tt = 0; tt < tn; tt++)\n'
'                                kv8_dequant_noscale(Vl8 + (kbase + t0 + tt)*kvd, vq + (size_t)tt*kvd, kvd);\n',
'                        if (m->kv8 || m->kv_tq)\n'
'                            for (int tt = 0; tt < tn; tt++)\n'
'                                q36_kv_dequant_noscale(Vl8 + (kbase + t0 + tt)*rb, vq + (size_t)tt*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
)

# ---------------------------------------------------------------- E10: flash V scale + row pick
edit(
'                                    if (m->kv8) p *= Vsc[kbase + t];\n'
'                                    lvs[li] += p;\n'
'                                    const float *vr = m->kv8 ? vq + (size_t)(t - t0)*kvd\n'
'                                                             : Vl_f + (kbase + t)*kvd;\n',
'                                    if (m->kv8 || m->kv_tq) p *= Vsc[kbase + t];\n'
'                                    lvs[li] += p;\n'
'                                    const float *vr = (m->kv8 || m->kv_tq) ? vq + (size_t)(t - t0)*kvd\n'
'                                                             : Vl_f + (kbase + t)*kvd;\n'
)

# ---------------------------------------------------------------- E11: decode staging pointer + K tile block
edit(
'            float *kq = m->kv8 ? m->kv8_stage + (int64_t)tid * (2 * KV8_TILE_ROWS * kvd) : NULL;\n'
'            if (m->kv8) {\n'
'                /* tile-staged: dequant KV8_TILE_ROWS K-rows at a time into per-thread\n'
'                 * staging (branch-free decode), then the same vectorized dot per row.\n'
'                 * Staging holds K then V halves: 2 * TILE_ROWS * kvd floats. */\n'
'                const uint8_t *K8 = (const uint8_t*)Kl;\n'
'                const uint8_t *V8 = (const uint8_t*)Vl;\n'
'                const float *Ksc = mr ? mr[s].seq->Kscale[layer] : m->Kscale[layer];\n'
'                const float *Vsc = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'                int64_t base = (int64_t)kvh*m->max_t;\n'
'                for (int t0 = 0; t0 <= qpos; t0 += KV8_TILE_ROWS) {\n'
'                    int tn = qpos - t0 + 1; if (tn > KV8_TILE_ROWS) tn = KV8_TILE_ROWS;\n'
'                    for (int t = 0; t < tn; t++)\n'
'                        kv8_dequant_noscale(K8 + (base + t0 + t)*kvd,\n'
'                                            kq + (int64_t)t*kvd, kvd);\n',
'            float *kq = (m->kv8 || m->kv_tq) ? m->kv8_stage + (int64_t)tid * (2 * KV8_TILE_ROWS * kvd) : NULL;\n'
'            if (m->kv8 || m->kv_tq) {\n'
'                /* tile-staged: dequant KV8_TILE_ROWS K-rows at a time into per-thread\n'
'                 * staging (branch-free decode), then the same vectorized dot per row.\n'
'                 * Staging holds K then V halves: 2 * TILE_ROWS * kvd floats.\n'
'                 * KV_TQ: the row stride is the packed kvq_row_b bytes and decode\n'
'                 * yields the unit direction; the radius rides Kscale/Vscale\n'
'                 * exactly like the KV8 per-row scale. */\n'
'                const int rb = m->kv_tq ? m->kvq_row_b : kvd;\n'
'                const uint8_t *K8 = (const uint8_t*)Kl;\n'
'                const uint8_t *V8 = (const uint8_t*)Vl;\n'
'                const float *Ksc = mr ? mr[s].seq->Kscale[layer] : m->Kscale[layer];\n'
'                const float *Vsc = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'                int64_t base = (int64_t)kvh*m->max_t;\n'
'                for (int t0 = 0; t0 <= qpos; t0 += KV8_TILE_ROWS) {\n'
'                    int tn = qpos - t0 + 1; if (tn > KV8_TILE_ROWS) tn = KV8_TILE_ROWS;\n'
'                    for (int t = 0; t < tn; t++)\n'
'                        q36_kv_dequant_noscale(K8 + (base + t0 + t)*rb,\n'
'                                               kq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
)

# ---------------------------------------------------------------- E12: decode V tile block
edit(
'            if (m->kv8) {\n'
'                const uint8_t *V8 = (const uint8_t*)Vl;\n'
'                const float *Vsc = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'                float *vq = kq + (int64_t)KV8_TILE_ROWS*kvd;   /* V half of staging */\n'
'                int64_t base = (int64_t)kvh*m->max_t;\n'
'                for (int t0 = 0; t0 <= qpos; t0 += KV8_TILE_ROWS) {\n'
'                    int tn = qpos - t0 + 1; if (tn > KV8_TILE_ROWS) tn = KV8_TILE_ROWS;\n'
'                    for (int t = 0; t < tn; t++)\n'
'                        kv8_dequant_noscale(V8 + (base + t0 + t)*kvd,\n'
'                                            vq + (int64_t)t*kvd, kvd);\n',
'            if (m->kv8 || m->kv_tq) {\n'
'                const uint8_t *V8 = (const uint8_t*)Vl;\n'
'                const float *Vsc = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'                float *vq = kq + (int64_t)KV8_TILE_ROWS*kvd;   /* V half of staging */\n'
'                const int rb = m->kv_tq ? m->kvq_row_b : kvd;\n'
'                int64_t base = (int64_t)kvh*m->max_t;\n'
'                for (int t0 = 0; t0 <= qpos; t0 += KV8_TILE_ROWS) {\n'
'                    int tn = qpos - t0 + 1; if (tn > KV8_TILE_ROWS) tn = KV8_TILE_ROWS;\n'
'                    for (int t = 0; t < tn; t++)\n'
'                        q36_kv_dequant_noscale(V8 + (base + t0 + t)*rb,\n'
'                                               vq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
)

# ---------------------------------------------------------------- E13: q36_seq_alloc scale calloc
edit(
'    if (m->kv8) { q->Kscale = calloc((size_t)n, sizeof(float *)); q->Vscale = calloc((size_t)n, sizeof(float *)); }\n',
'    if (m->kv8 || m->kv_tq) { q->Kscale = calloc((size_t)n, sizeof(float *)); q->Vscale = calloc((size_t)n, sizeof(float *)); }\n'
)

# ---------------------------------------------------------------- E14: q36_seq_alloc ok check
edit(
'    int ok = q->K && q->V && (!m->kv8 || (q->Kscale && q->Vscale)) && q->DN_rec && q->DN_conv && kv_prefix_alloc(&q->kvp, m->kv_cap);\n',
'    int ok = q->K && q->V && (!(m->kv8 || m->kv_tq) || (q->Kscale && q->Vscale)) && q->DN_rec && q->DN_conv && kv_prefix_alloc(&q->kvp, m->kv_cap);\n'
)

# ---------------------------------------------------------------- E15: q36_seq_alloc per-layer TQ branch
edit(
'        if (c->is_attn[i]) {\n'
'            size_t kv = (size_t)c->kv_heads * m->kv_cap * c->k_head_dim;\n'
'            if (m->kv8) {\n'
'                q->K[i] = malloc(kv); q->V[i] = malloc(kv);   /* 1 B/element, fp8 */\n',
'        if (c->is_attn[i]) {\n'
'            size_t kv = (size_t)c->kv_heads * m->kv_cap * c->k_head_dim;\n'
'            if (m->kv_tq) {\n'
'                size_t kvb = (size_t)c->kv_heads * m->kv_cap * (size_t)m->kvq_row_b;\n'
'                q->K[i] = malloc(kvb); q->V[i] = malloc(kvb);   /* packed B/row, codec-dependent */\n'
'                q->Kscale[i] = malloc((size_t)c->kv_heads * m->kv_cap * sizeof(float));\n'
'                q->Vscale[i] = malloc((size_t)c->kv_heads * m->kv_cap * sizeof(float));\n'
'                ok = q->K[i] && q->V[i] && q->Kscale[i] && q->Vscale[i];\n'
'            } else if (m->kv8) {\n'
'                q->K[i] = malloc(kv); q->V[i] = malloc(kv);   /* 1 B/element, fp8 */\n'
)

# ---------------------------------------------------------------- E16: model kv-grow scale calloc
edit(
'    if (m->kv8) { m->Kscale = calloc((size_t)c->n_layers, sizeof(float*)); m->Vscale = calloc((size_t)c->n_layers, sizeof(float*)); }\n',
'    if (m->kv8 || m->kv_tq) { m->Kscale = calloc((size_t)c->n_layers, sizeof(float*)); m->Vscale = calloc((size_t)c->n_layers, sizeof(float*)); }\n'
)

# ---------------------------------------------------------------- E17: model kv-grow TQ branch (stride-correct row copies)
edit(
'        if (c->is_attn[i]){\n'
'            int64_t kvd = c->k_head_dim;\n'
'            if (m->kv8) {\n',
'        if (c->is_attn[i]){\n'
'            int64_t kvd = c->k_head_dim;\n'
'            if (m->kv_tq) {\n'
'                int64_t rb = m->kvq_row_b;   /* packed-row stride (bytes) */\n'
'                m->K[i] = falloc((int64_t)c->kv_heads * m->max_t * rb);\n'
'                m->V[i] = falloc((int64_t)c->kv_heads * m->max_t * rb);\n'
'                m->Kscale[i] = falloc((int64_t)c->kv_heads * m->max_t);\n'
'                m->Vscale[i] = falloc((int64_t)c->kv_heads * m->max_t);\n'
'                if (keep > 0 && oldK && oldK[i] && oldV[i] && oldKscale && oldKscale[i])\n'
'                    for (int h = 0; h < c->kv_heads; h++){\n'
'                        memcpy((uint8_t*)m->K[i] + (int64_t)h*m->max_t*rb,\n'
'                               (uint8_t*)oldK[i] + (int64_t)h*old_stride*rb, (size_t)keep*rb);\n'
'                        memcpy((uint8_t*)m->V[i] + (int64_t)h*m->max_t*rb,\n'
'                               (uint8_t*)oldV[i] + (int64_t)h*old_stride*rb, (size_t)keep*rb);\n'
'                        memcpy(m->Kscale[i] + (int64_t)h*m->max_t, oldKscale[i] + (int64_t)h*old_stride, (size_t)keep*sizeof(float));\n'
'                        memcpy(m->Vscale[i] + (int64_t)h*m->max_t, oldVscale[i] + (int64_t)h*old_stride, (size_t)keep*sizeof(float));\n'
'                    }\n'
'            } else if (m->kv8) {\n'
)

# ---------------------------------------------------------------- E18: model kv-grow NULL-else
edit(
'        } else { m->K[i] = NULL; m->V[i] = NULL; if (m->kv8){ m->Kscale[i]=NULL; m->Vscale[i]=NULL; } }\n',
'        } else { m->K[i] = NULL; m->V[i] = NULL; if (m->kv8 || m->kv_tq){ m->Kscale[i]=NULL; m->Vscale[i]=NULL; } }\n'
)

# ---------------------------------------------------------------- E19: kv8_stage alloc (staging is f32 rows — same size for TQ)
edit(
'    m->kv8_stage = m->kv8 ? falloc((int64_t)m->attn_sc_thr * 2 * KV8_TILE_ROWS * c->k_head_dim) : NULL;\n',
'    m->kv8_stage = (m->kv8 || m->kv_tq) ? falloc((int64_t)m->attn_sc_thr * 2 * KV8_TILE_ROWS * c->k_head_dim) : NULL;\n'
)

# ---------------------------------------------------------------- E20: env parse (after model_init) + dedup the doubled KV8 banner
edit(
'    m.kv8 = getenv("KV8") && atoi(getenv("KV8")) == 1;   /* after model_init: it memsets the struct */\n'
'    if (m.kv8) coli_fp8_lut_init();\n'
'    if (m.kv8) fprintf(stderr, "[qwen36] KV8=1: KV rows stored fp8-e4m3 + per-row f32 scale (lossy, opt-in)\\n");\n'
'    if (m.kv8) fprintf(stderr, "[qwen36] KV8=1: KV rows stored fp8-e4m3 + per-row f32 scale (lossy, opt-in)\\n");\n',
'    m.kv8 = getenv("KV8") && atoi(getenv("KV8")) == 1;   /* after model_init: it memsets the struct */\n'
'    if (getenv("KV_TQ")) {\n'
'        int tq_b = atoi(getenv("KV_TQ"));\n'
'        if (tq_b >= 2 && tq_b <= 8) {\n'
'            m.kv_tq = 1; m.tq_bits = tq_b;\n'
'            m.kv8 = 0;   /* mutually exclusive with KV8; KV_TQ wins */\n'
'            m.tq_codec = getenv("KV_TQ_CODEC") ? atoi(getenv("KV_TQ_CODEC")) : 1;   /* 1 = rotated int4 (default, cheap decode), 0 = PolarQuant (RAM-extreme) */\n'
'            if (m.tq_codec != 0 && m.tq_codec != 1) m.tq_codec = 1;\n'
'            if (m.c.k_head_dim < 2 || m.c.k_head_dim > 2048 || (m.c.k_head_dim & (m.c.k_head_dim - 1))\n'
'                || m.c.k_head_dim != m.c.v_head_dim) {\n'
'                fprintf(stderr, "[qwen36] KV_TQ: k_head_dim %d (v %d) unsupported — need equal power of two <= 2048; refusing\\n",\n'
'                        m.c.k_head_dim, m.c.v_head_dim);\n'
'                exit(1);\n'
'            }\n'
'            m.kvq_row_b = coli_kvq_row_bytes(m.c.k_head_dim, m.tq_bits, m.tq_codec);\n'
'            fprintf(stderr, "[qwen36] KV_TQ=%d codec=%d (%s): KV rows packed %d B + f32 radius/row (vs fp8 %d B, f32 %d B) (lossy, opt-in)\\n",\n'
'                    m.tq_bits, m.tq_codec, m.tq_codec ? "rot-int4" : "PolarQuant",\n'
'                    m.kvq_row_b, m.c.k_head_dim, m.c.k_head_dim * 4);\n'
'        } else fprintf(stderr, "[qwen36] KV_TQ=%d ignored (supported: 2..8)\\n", tq_b);\n'
'    }\n'
'    if (m.kv8) coli_fp8_lut_init();\n'
'    if (m.kv8) fprintf(stderr, "[qwen36] KV8=1: KV rows stored fp8-e4m3 + per-row f32 scale (lossy, opt-in)\\n");\n'
)

def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
    src = open(path, "r", encoding="utf-8", newline="").read()
    if "\\r\\n" in repr(src[:2000]) and "\r\n" in src[:2000]:
        sys.exit("refusing: file appears to be CRLF — anchors are LF; inspect manually")
    if "kv_tq" in src or "kvq_row_b" in src:
        sys.exit("refusing: qwen36.c already carries KV_TQ markers (idempotence guard)")
    for i, (old, new) in enumerate(EDITS):
        n = src.count(old)
        if n != 1:
            sys.exit("E%02d: anchor count=%d (need 1): %.80r" % (i + 1, n, old))
        src = src.replace(old, new, 1)
    bak = path + ".pretq"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print("backup -> %s" % bak)
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    print("applied %d edits -> %s" % (len(EDITS), path))

if __name__ == "__main__":
    main()
