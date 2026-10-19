#!/usr/bin/env python3
"""patch_gqa_absorb.py -- Phase 3a: dequant each KV tile ONCE per GQA group.

The decode branch (S <= 8, flash-off path) walks (hh, s) pairs and
dequantizes the same K/V tiles once per QUERY HEAD -- q_per_kv times more
dequant work than the data needs. Under f32 KV there is no dequant (the dot
reads f32 directly) so nobody noticed; under KV8 -- and especially KV_TQ,
whose decode includes a Hadamard unrotate plus codec -- this redundancy is
most of the decode cost of the quantized tiers (measured on synth_qwen27
@16k: f32 46.9 tok/s, KV8 26.6, KV_TQ4 5.3; the 27B has q_per_kv = 5).

The absorbed branch restructures the loop to (kvh, s): each K tile and V
tile is dequantized once into the per-thread staging and serves every head
of its group. The arithmetic per (head, t) is UNCHANGED -- same dots over
the same dequantized values, same softmax over the same score row, same
t-ascending V accumulation -- so the output is bit-identical to the
per-head loop; only the dequant count drops by q_per_kv x.

Scratch:
  m->gqa_sc  [attn_sc_thr][q_per_kv][kv_cap]  per-head score rows of a
             group (allocated beside attn_sc in the kv-grow path; NULL on
             adapters that never grow -- those fall back to the original
             loop through the m->gqa_sc guard).
  kv8_stage  grows by q_per_kv*kvd floats per thread: the head-group
             output accumulators live in that tail.

No knobs: pure restructure of the quantized-KV decode path, active
whenever KV8 or KV_TQ is on. f32 path untouched. Knob-off engines see
zero code-path change (the branch guard falls through).

Chain position: applies to the post-Phase-2 engine (qwen36.c at
phase-2b-5, md5 d6b3ff1c...). Anchored byte-exact edits, backup
qwen36.c.pregqa. Refuses to run twice.
Usage: python3 patch_gqa_absorb.py <path-to-qwen36.c>
"""
import os, shutil, sys

EDITS = []
def edit(old, new): EDITS.append((old, new))

# ---------------------------------------------------------------- E1: struct field
edit(
'    float *attn_sc;            /* [attn_sc_thr * kv_cap] score rows, one per thread */\n'
'    int attn_sc_thr;\n',
'    float *attn_sc;            /* [attn_sc_thr * kv_cap] score rows, one per thread */\n'
'    int attn_sc_thr;\n'
'    float *gqa_sc;             /* [attn_sc_thr * q_per_kv * kv_cap] GQA-absorbed decode: one score row per head of a group (KV8/KV_TQ) */\n'
)

# ---------------------------------------------------------------- E2a: grow-site free
edit(
'    free(m->attn_sc);\n'
'    m->attn_sc_thr = 1;\n'
'#ifdef _OPENMP\n',
'    free(m->attn_sc);\n'
'    free(m->gqa_sc);\n'
'    m->attn_sc_thr = 1;\n'
'#ifdef _OPENMP\n'
)

# ---------------------------------------------------------------- E2b: grow-site alloc
edit(
'    m->attn_sc = falloc((int64_t)m->attn_sc_thr * m->max_t);\n'
'    free(m->kv8_stage);\n'
'    m->kv8_stage = (m->kv8 || m->kv_tq) ? falloc((int64_t)m->attn_sc_thr * 2 * KV8_TILE_ROWS * c->k_head_dim) : NULL;\n',
'    m->attn_sc = falloc((int64_t)m->attn_sc_thr * m->max_t);\n'
'    free(m->kv8_stage);\n'
'    m->kv8_stage = (m->kv8 || m->kv_tq) ? falloc((int64_t)m->attn_sc_thr * (2 * KV8_TILE_ROWS + c->q_heads / c->kv_heads) * c->k_head_dim) : NULL;\n'
'    /* GQA-absorbed decode: one score row per head of a group, per thread;\n'
'     * q_per_kv rows keep a whole group\'s scores live across the tile walk. */\n'
'    m->gqa_sc = (m->kv8 || m->kv_tq) ? falloc((int64_t)m->attn_sc_thr * (c->q_heads / c->kv_heads) * m->max_t) : NULL;\n'
)

# ---------------------------------------------------------------- E3: model_free
edit(
'    free(model->attn_sc);\n',
'    free(model->attn_sc);\n'
'    free(model->gqa_sc);\n'
)

# ---------------------------------------------------------------- E4: the absorbed decode branch
edit(
'    } else {\n'
'    #pragma omp parallel for collapse(2) schedule(static)\n'
'    for (int hh = 0; hh < H; hh++) {\n'
'        for (int s = 0; s < S; s++) {\n',
'    } else if ((m->kv8 || m->kv_tq) && m->gqa_sc) {\n'
'        /* GQA-absorbed decode tiles: dequantize each K/V tile ONCE per\n'
'         * (kvh, s) and serve every head of its group. The per-head loop\n'
'         * below pays the same dequant q_per_kv times -- on the real 27B\n'
'         * (5 q heads per kv head) that is most of the KV8/KV_TQ decode\n'
'         * cost. The arithmetic per (head, t) is unchanged (same dots,\n'
'         * same softmax row, same t-ascending V accumulation), so the\n'
'         * output is bit-identical, cheaper. */\n'
'        #pragma omp parallel for collapse(2) schedule(static)\n'
'        for (int kvh = 0; kvh < KV; kvh++) {\n'
'        for (int s = 0; s < S; s++) {\n'
'            int qpos = mr ? mr[s].pos : pos_base + s;\n'
'            const float *Kl = mr ? mr[s].seq->K[layer] : m->K[layer], *Vl = mr ? mr[s].seq->V[layer] : m->V[layer];\n'
'            const uint8_t *K8 = (const uint8_t*)Kl, *V8 = (const uint8_t*)Vl;\n'
'            const float *Ksc = mr ? mr[s].seq->Kscale[layer] : m->Kscale[layer];\n'
'            const float *Vsc = mr ? mr[s].seq->Vscale[layer] : m->Vscale[layer];\n'
'            const int rb = m->kv_tq ? m->kvq_row_b : kvd;\n'
'            int64_t base = (int64_t)kvh*m->max_t;\n'
'            int tid = 0;\n'
'#ifdef _OPENMP\n'
'            tid = omp_get_thread_num();\n'
'#endif\n'
'            float *scg = m->gqa_sc + (int64_t)tid * ((int64_t)q_per_kv * m->kv_cap);\n'
'            float *kq = m->kv8_stage + (int64_t)tid * ((2 * KV8_TILE_ROWS + q_per_kv) * kvd);\n'
'            float *vq = kq + (int64_t)KV8_TILE_ROWS * kvd;\n'
'            float *cxg = vq + (int64_t)KV8_TILE_ROWS * kvd;   /* q_per_kv*kvd tail of the grown staging */\n'
'            for (int g = 0; g < q_per_kv; g++)\n'
'                for (int dd = 0; dd < kvd; dd++) cxg[(int64_t)g*kvd + dd] = 0.f;\n'
'            for (int t0 = 0; t0 <= qpos; t0 += KV8_TILE_ROWS) {\n'
'                int tn = qpos - t0 + 1; if (tn > KV8_TILE_ROWS) tn = KV8_TILE_ROWS;\n'
'                for (int t = 0; t < tn; t++)\n'
'                    q36_kv_dequant_noscale(K8 + (base + t0 + t)*rb,\n'
'                                           kq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
'                for (int g = 0; g < q_per_kv; g++) {\n'
'                    const float *qv = query + ((int64_t)s*H + (kvh*q_per_kv + g))*hd;\n'
'                    float *sc = scg + (int64_t)g * m->kv_cap;\n'
'                    for (int t = 0; t < tn; t++)\n'
'                        sc[t0 + t] = dot_f32_lanes(qv, kq + (int64_t)t*kvd, kvd)\n'
'                                     * (Ksc[base + t0 + t] * scale);\n'
'                }\n'
'            }\n'
'            for (int g = 0; g < q_per_kv; g++)\n'
'                softmax_row(scg + (int64_t)g * m->kv_cap, qpos + 1);\n'
'            for (int t0 = 0; t0 <= qpos; t0 += KV8_TILE_ROWS) {\n'
'                int tn = qpos - t0 + 1; if (tn > KV8_TILE_ROWS) tn = KV8_TILE_ROWS;\n'
'                for (int t = 0; t < tn; t++)\n'
'                    q36_kv_dequant_noscale(V8 + (base + t0 + t)*rb,\n'
'                                           vq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
'                for (int t = 0; t < tn; t++) {\n'
'                    const float *vrow = vq + (int64_t)t*kvd;\n'
'                    for (int g = 0; g < q_per_kv; g++) {\n'
'                        float a = scg[(int64_t)g*m->kv_cap + t0 + t] * Vsc[base + t0 + t];\n'
'                        float *cx = cxg + (int64_t)g*kvd;\n'
'                        for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];\n'
'                    }\n'
'                }\n'
'            }\n'
'            for (int g = 0; g < q_per_kv; g++)\n'
'                memcpy(ctx + ((int64_t)s*H + (kvh*q_per_kv + g))*hd,\n'
'                       cxg + (int64_t)g*kvd, (size_t)kvd*sizeof(float));\n'
'        }\n'
'        }\n'
'    } else {\n'
'    #pragma omp parallel for collapse(2) schedule(static)\n'
'    for (int hh = 0; hh < H; hh++) {\n'
'        for (int s = 0; s < S; s++) {\n'
)

def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
    src = open(path, "r", encoding="utf-8", newline="").read()
    if "gqa_sc" in src:
        sys.exit("refusing: qwen36.c already carries gqa_sc markers (idempotence guard)")
    for i, (old, new) in enumerate(EDITS):
        n = src.count(old)
        if n != 1:
            sys.exit("E%02d: anchor count=%d (need 1): %.80r" % (i + 1, n, old))
        src = src.replace(old, new, 1)
    bak = path + ".pregqa"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print("backup -> %s" % bak)
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    print("applied %d edits -> %s" % (len(EDITS), path))

if __name__ == "__main__":
    main()
