#!/usr/bin/env python3
"""patch_scan_decay.py -- Phase 3b-scan: two-tier deterministic decode scan.

Design contract: docs/SCAN_DECAY_QWEN36.md. After 3a (dequant once per GQA
group) and 3b-rot/prefill (no per-row FWHT), decode attention still scans
all qpos+1 KV rows per token -- O(T), 57.65 ms/tok at the 68.4k fixture.
This patch adds the contract's two-tier scan to the ABSORBED decode branch
(KV8/KV_TQ arms, incl. the TQ-absorb variant):

  tier 1 (always): sink blocks (first Q36_SCAN_SINK rows) + local window
                   (last Q36_SCAN_WIN rows)
  tier 2:          middle 128-row blocks in descending salience until the
                   cumulative middle-mass share reaches 1 - Q36_SCAN_EPS

Salience = attention mass per (layer, kv-head, block), accumulated where
scores already exist: the flash prefill inner loop (pos_base==0, the true
prefill) and the decode SETTLE row (s==0 -- the one verify row that always
stands, which removes the snapshot/rollback interaction entirely; contract
amendment, documented in the header and the doc). The table is uint64
fixed-point (x 2^14) so schedule(dynamic) grouping cannot make it vary run
to run: integer sums are order-associative, hence bit-reproducible scan
sets and ids (gate 1).

Q36_SCAN=full (the default) keeps the pre-scan loops verbatim; f32 decode,
mux/serve rows (mr) and adapters are untouched (full scan). Flash seeding
is the only change the full path can reach -- it is gated on the same knob.

New header: engine/scan_decay.h (knobs, builder, report). Backup
qwen36.c.prescan. Refuses to run twice.
Usage: python3 patch_scan_decay.py <path-to-qwen36.c>
"""
import os, shutil, sys

EDITS = []
def edit(old, new): EDITS.append((old, new))

# ------------------------------------------------------------- E1: include + helpers site
edit(
'static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {\n',
'#include "scan_decay.h"   /* Phase 3b-scan: two-tier deterministic decode scan (knobs, builder, report) */\n'
'\n'
'static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {\n'
)

# ------------------------------------------------------------- E2: Model fields
edit(
'    float *gqa_sc;             /* [attn_sc_thr * q_per_kv * kv_cap] GQA-absorbed decode: one score row per head of a group (KV8/KV_TQ) */\n'
'    double dense_load_s;\n',
'    float *gqa_sc;             /* [attn_sc_thr * q_per_kv * kv_cap] GQA-absorbed decode: one score row per head of a group (KV8/KV_TQ) */\n'
'    uint64_t *sal;             /* 3b-scan: fixed-point (x2^14) attention mass per [layer][kv_head][128-row block] */\n'
'    int sal_blocks;            /*   block columns of sal; grows with max_t, contents preserved across grows */\n'
'    int *scan_rows;            /*   per-thread scratch: [thr][kv_cap row-ids | sbcap block-ids | sbcap sel bytes] */\n'
'    double dense_load_s;\n'
)

# ------------------------------------------------------------- E3: grow site (alloc + preserve + scratch)
edit(
'    m->gqa_sc = (m->kv8 || m->kv_tq) ? falloc((int64_t)m->attn_sc_thr * (c->q_heads / c->kv_heads) * m->max_t) : NULL;\n'
'    m->kv_cap = m->max_t;\n',
'    m->gqa_sc = (m->kv8 || m->kv_tq) ? falloc((int64_t)m->attn_sc_thr * (c->q_heads / c->kv_heads) * m->max_t) : NULL;\n'
'    m->kv_cap = m->max_t;\n'
'    /* 3b-scan: salience table [n_layers][kv_heads][sal_blocks] uint64. It\n'
'     * grows with the context like everything here, but its contents must\n'
'     * SURVIVE a grow (the prefill seeds it once), so a grow copies the old\n'
'     * per-(layer,kvh) rows forward. Fixed-point, not f32: the flash path\n'
'     * seeds under schedule(dynamic) and integer sums are order-associative,\n'
'     * so the merged table (and the scan set, and the ids) is bit-exact. */\n'
'    if (q36_scan_two() && (m->kv8 || m->kv_tq)) {\n'
'        int nb = (m->max_t + 127) >> 7;\n'
'        if (!m->sal) {\n'
'            m->sal = calloc((size_t)((int64_t)c->n_layers * c->kv_heads) * nb, sizeof(uint64_t));\n'
'            if (!m->sal) { fprintf(stderr, "[scan] OOM sal\\n"); exit(1); }\n'
'            m->sal_blocks = nb;\n'
'        } else if (nb > m->sal_blocks) {\n'
'            int64_t cells = (int64_t)c->n_layers * c->kv_heads;\n'
'            uint64_t *ns = calloc((size_t)(cells * nb), sizeof(uint64_t));\n'
'            if (!ns) { fprintf(stderr, "[scan] OOM sal grow\\n"); exit(1); }\n'
'            for (int64_t i = 0; i < cells; i++)\n'
'                memcpy(ns + i * nb, m->sal + i * m->sal_blocks,\n'
'                       (size_t)m->sal_blocks * sizeof(uint64_t));\n'
'            free(m->sal); m->sal = ns; m->sal_blocks = nb;\n'
'        }\n'
'        int sbcap = (m->kv_cap >> 7) + 8;\n'
'        free(m->scan_rows);\n'
'        m->scan_rows = calloc((size_t)m->attn_sc_thr * (m->kv_cap + 2 * sbcap), sizeof(int));\n'
'        if (!m->scan_rows) { fprintf(stderr, "[scan] OOM scratch\\n"); exit(1); }\n'
'    } else if (m->sal || m->scan_rows) {\n'
'        free(m->sal); m->sal = NULL; m->sal_blocks = 0;\n'
'        free(m->scan_rows); m->scan_rows = NULL;\n'
'    }\n'
)

# ------------------------------------------------------------- E4: model_free
edit(
'    free(model->gqa_sc);\n',
'    free(model->gqa_sc);\n'
'    free(model->sal); free(model->scan_rows);\n'
)

# ------------------------------------------------------------- E5a: scan walk before the K loop
edit(
'            if (absorb)\n'
'                for (int g = 0; g < q_per_kv; g++)\n'
'                    coli_tq_rotate(query + ((int64_t)s*H + (kvh*q_per_kv + g))*hd,\n'
'                                   qrg + (size_t)g*kvd, kvd, COLI_TQ_SEED);\n',
'            if (absorb)\n'
'                for (int g = 0; g < q_per_kv; g++)\n'
'                    coli_tq_rotate(query + ((int64_t)s*H + (kvh*q_per_kv + g))*hd,\n'
'                                   qrg + (size_t)g*kvd, kvd, COLI_TQ_SEED);\n'
'            /* 3b-scan: two-tier walk (docs/SCAN_DECAY_QWEN36.md). The scan\n'
'             * set is a deterministic function of the fixed-point salience\n'
'             * table; skipped rows contribute exactly 0 mass because the\n'
'             * softmax renormalizes over the scanned slots only (the\n'
'             * documented approximation, hence the real-weights gates). */\n'
'            int scan = (q36_scan_two() && !mr && m->sal) ? 1 : 0;\n'
'            int *srows = NULL, *sbl = NULL; unsigned char *ssel = NULL;\n'
'            if (scan) {\n'
'                int sbcap = (m->kv_cap >> 7) + 8;\n'
'                srows = m->scan_rows + (int64_t)tid * (m->kv_cap + 2 * sbcap);\n'
'                sbl = srows + m->kv_cap;\n'
'                ssel = (unsigned char*)(sbl + sbcap);\n'
'                int nbl = q36_scan_build(m, layer, kvh, qpos, ssel, sbl);\n'
'                int nslot = 0;\n'
'                for (int bi = 0; bi < nbl; bi++) {\n'
'                    int t0 = sbl[bi] << 7;\n'
'                    int tn = qpos - t0 + 1; if (tn > 128) tn = 128;\n'
'                    if (tn <= 0) continue;\n'
'                    for (int t = 0; t < tn; t++) {\n'
'                        if (absorb) q36_q4_rot_noscale(K8 + (base + t0 + t)*rb, kq + (int64_t)t*kvd, kvd);\n'
'                        else q36_kv_dequant_noscale(K8 + (base + t0 + t)*rb,\n'
'                                                    kq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
'                        srows[nslot + t] = t0 + t;\n'
'                    }\n'
'                    for (int g = 0; g < q_per_kv; g++) {\n'
'                        const float *qvs = absorb ? qrg + (size_t)g*kvd\n'
'                                                  : query + ((int64_t)s*H + (kvh*q_per_kv + g))*hd;\n'
'                        float *sc = scg + (int64_t)g * m->kv_cap;\n'
'                        for (int t = 0; t < tn; t++)\n'
'                            sc[nslot + t] = dot_f32_lanes(qvs, kq + (int64_t)t*kvd, kvd)\n'
'                                          * (Ksc[base + t0 + t] * scale);\n'
'                    }\n'
'                    nslot += tn;\n'
'                }\n'
'                for (int g = 0; g < q_per_kv; g++)\n'
'                    softmax_row(scg + (int64_t)g * m->kv_cap, nslot);\n'
'                if (s == 0 || g_q36_scan_prefill) {   /* settle row (the one\n'
'                                  row of a verify that always stands) or any\n'
'                                  prefill chunk row -- no snapshot/rollback\n'
'                                  needed either way */\n'
'                    uint64_t *salrow = Q36_SAL(m, layer, kvh);\n'
'                    for (int g = 0; g < q_per_kv; g++) {\n'
'                        float *sc = scg + (int64_t)g * m->kv_cap;\n'
'                        for (int i = 0; i < nslot; i++)\n'
'                            salrow[srows[i] >> 7] += (uint64_t)(sc[i] * (float)Q36_SCAN_SCALE);\n'
'                    }\n'
'                }\n'
'                int slot = 0;\n'
'                for (int bi = 0; bi < nbl; bi++) {\n'
'                    int t0 = sbl[bi] << 7;\n'
'                    int tn = qpos - t0 + 1; if (tn > 128) tn = 128;\n'
'                    if (tn <= 0) continue;\n'
'                    for (int t = 0; t < tn; t++) {\n'
'                        if (absorb) q36_q4_rot_noscale(V8 + (base + t0 + t)*rb, vq + (int64_t)t*kvd, kvd);\n'
'                        else q36_kv_dequant_noscale(V8 + (base + t0 + t)*rb,\n'
'                                                    vq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);\n'
'                    }\n'
'                    for (int t = 0; t < tn; t++) {\n'
'                        const float *vrow = vq + (int64_t)t*kvd;\n'
'                        int row = srows[slot + t];\n'
'                        for (int g = 0; g < q_per_kv; g++) {\n'
'                            float a = scg[(int64_t)g*m->kv_cap + slot + t] * Vsc[base + row];\n'
'                            float *cx = cxg + (int64_t)g*kvd;\n'
'                            for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];\n'
'                        }\n'
'                    }\n'
'                    slot += tn;\n'
'                }\n'
'            }\n'
'            if (!scan) {\n'
)

# ------------------------------------------------------------- E5b: close the full-scan wrap
edit(
'            if (absorb)\n'
'                for (int g = 0; g < q_per_kv; g++)\n'
'                    coli_tq_unrotate(cxg + (size_t)g*kvd, kvd, COLI_TQ_SEED);\n',
'            }   /* !scan: the 3a/3b loops verbatim -- Q36_SCAN=full identity path */\n'
'            if (absorb)\n'
'                for (int g = 0; g < q_per_kv; g++)\n'
'                    coli_tq_unrotate(cxg + (size_t)g*kvd, kvd, COLI_TQ_SEED);\n'
)

# ------------------------------------------------------------- E6a: flash seeding scaffolding
edit(
'        #pragma omp parallel\n'
'        {\n'
'            float *sc  = malloc((size_t)QB*q_per_kv*TB*sizeof(float));\n',
'        /* 3b-scan prefill seeding: per-thread fixed-point partials merged\n'
'         * after the region in thread order (integer sums are exact, so the\n'
'         * dynamic schedule cannot make the table vary run to run). Only\n'
'         * the true prefill (pos_base == 0) seeds. */\n'
'        int scols = m->sal ? m->sal_blocks : 0;\n'
'        int sthr = 1;\n'
'#ifdef _OPENMP\n'
'        sthr = omp_get_max_threads();\n'
'#endif\n'
'        uint64_t *sseed = (q36_scan_two() && scols > 0 && g_q36_scan_prefill)\n'
'                          ? calloc((size_t)sthr * KV * scols, sizeof(uint64_t)) : NULL;\n'
'        #pragma omp parallel\n'
'        {\n'
'            int f_tid = 0;\n'
'#ifdef _OPENMP\n'
'            f_tid = omp_get_thread_num();\n'
'#endif\n'
'            float *sc  = malloc((size_t)QB*q_per_kv*TB*sizeof(float));\n'
)

# ------------------------------------------------------------- E6b: split pw from p
edit(
'                                for (int t = t0; t <= t_last; t++) {\n'
'                                    float p = expf(sc[(size_t)li*TB + (t - t0)] - m2);\n',
'                                for (int t = t0; t <= t_last; t++) {\n'
'                                    float pw = expf(sc[(size_t)li*TB + (t - t0)] - m2);\n'
'                                    float p = pw;   /* pw: un-V-scaled mass, for the scan seeding */\n'
)

# ------------------------------------------------------------- E6c: the seed add
edit(
'                                    lvs[li] += p;\n',
'                                    lvs[li] += p;\n'
'                                    if (sseed) sseed[((int64_t)f_tid*KV + kvh)*scols + (t >> 7)]\n'
'                                                   += (uint64_t)(pw * (float)Q36_SCAN_SCALE);\n'
)

# ------------------------------------------------------------- E6d: fixed-order merge after the region
edit(
'            free(sc); free(acc); free(mxv); free(lvs); free(kq); free(vq);\n'
'        }\n',
'            free(sc); free(acc); free(mxv); free(lvs); free(kq); free(vq);\n'
'        }\n'
'        if (sseed) {   /* thread-order merge; integer sums keep it bit-exact */\n'
'            for (int i = 0; i < sthr; i++)\n'
'                for (int64_t j = 0; j < (int64_t)KV*scols; j++)\n'
'                    m->sal[(int64_t)layer*c->kv_heads*scols + j] += sseed[(int64_t)i*KV*scols + j];\n'
'            free(sseed);\n'
'        }\n'
)

# ------------------------------------------------------------- E8: mark the prefill driver
edit(
'        if (c->is_attn[i]) {\n'
'            attention(m, l, i, nrm, S, pos_base, tmp);\n'
'            tm_add(S, 1, tm_now()-_t0);\n',
'        if (c->is_attn[i]) {\n'
'            g_q36_scan_prefill = 1;   /* 3b-scan: this exact call signature exists only in the chunked prefill driver -- flag it so the seeding covers every chunk, never a verify batch */\n'
'            attention(m, l, i, nrm, S, pos_base, tmp);\n'
'            g_q36_scan_prefill = 0;\n'
'            tm_add(S, 1, tm_now()-_t0);\n'
)

# ------------------------------------------------------------- E7: report at the generation tail
edit(
'    q36_spec_end(m, sp);\n',
'    q36_spec_end(m, sp);\n'
'    q36_scan_report(m);\n'
)

def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
    src = open(path, "r", encoding="utf-8", newline="").read()
    if "scan_decay.h" in src:
        sys.exit("refusing: qwen36.c already includes scan_decay.h (idempotence guard)")
    for i, (old, new) in enumerate(EDITS):
        n = src.count(old)
        if n != 1:
            sys.exit("E%02d: anchor count=%d (need 1): %.90r" % (i + 1, n, old))
        src = src.replace(old, new, 1)
    bak = path + ".prescan"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print("backup -> %s" % bak)
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    print("applied %d edits -> %s" % (len(EDITS), path))

if __name__ == "__main__":
    main()
