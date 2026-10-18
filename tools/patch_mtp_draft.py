#!/usr/bin/env python3
"""patch_mtp_draft.py — Phase 2b part 2: MTP DRAFTING + verify integration (dense head).

Adds, on top of patch_mtp.py part 1:
  - head forward on the S=1 path: own f32 K/V rows (rope_head_partial, qk-norm,
    GQA scores over the head's own rows, softmax_row, attn output gate, o_proj),
    y-ring per row position; deeper drafts feed the head's own hidden back in
    (qwen38 semantics)
  - hidden capture at step_ex's lm_head seam (the nlogits settled rows, ring of 8)
  - q36_mtp_propose(): drafts for pos+1.. from (embed(tok), hidden[pos-1]);
    Q36_MTP_DRAFTS caps (0 = gate picks); Q36_MTP_FORCE reuses the lookup oracle
  - source choice beside lookup via spec_gate_pick values (SPEC_SRC_MTP vs
    SPEC_SRC_LOOKUP); sp->cur_src replaces the hardcoded source in settle and
    the report; head-row rollback = kv_len trim in q36_spec_settle
  - report line [qwen36 MTP] scope: acceptance / drafts / verifies / gate

Gates (CI engine-build): greedy stdout byte-identical to stock for
Q36_MTP_DRAFTS in {0,1,2,3} with COLI_LOOKUP=0 COLI_SPEC_GATE=0 (random-weight
head => ~0 acceptance => every verify rejects => rollback exercised every
verify); report line present. MoE head + forced-accept oracle battery = part 3.

Anchored exact-byte edits, house style. Backup qwen36.c.predraft. Refuses twice.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "q36_mtp_propose" in src:
    raise SystemExit("patch_mtp_draft: already applied")
if "Q36MTPHead" not in src:
    raise SystemExit("patch_mtp_draft: part 1 (patch_mtp.py) must be applied first")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# 1. head state fields
edit("""    int64_t n_fc, n_qw, n_kw, n_vw, n_ow, n_gw, n_uw, n_dw;
} MTPHead;
""",
     """    int64_t n_fc, n_qw, n_kw, n_vw, n_ow, n_gw, n_uw, n_dw;
    int kv_len;                 /* standing + provisional head rows */
    float *hk, *hv, *yring;     /* own KV [KVH][max_t][HD] + head hidden per row */
    float *cap; int cap_pos[8]; /* captured model hidden, ring of 8 positions */
} MTPHead;
""")

# 2. head forward + propose + capture + settle, before the spec machinery
edit("""static void q36_spec_begin(Model *m, Q36Spec *sp, const int *prompt, int np) {
""",
     """/* ---- Phase 2b part 2: MTP head forward + drafting (dense head) ---- */
static int q36_spec_forced(const Q36Spec *sp, int k, int j, int pos, int V);

static void mtp_gemv(float *y, const float *x, const float *W, int I, int O) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        float s = 0;
        for (int i = 0; i < I; i++) s += x[i] * w[i];
        y[o] = s;
    }
}

static const float *q36_mtp_hidden_at(Model *m, int p) {
    if (p < 0 || !m->mtp) return NULL;
    MTPHead *h = m->mtp; int slot = p & 7;
    return h->cap_pos[slot] == p + 1 ? h->cap + (int64_t)slot * m->c.hidden : NULL;
}

static void q36_mtp_capture(Model *m, const float *x, int S, int pos_base, int nlogits) {
    if (!g_q36_mtp_on || !m->mtp || !m->mtp->loaded) return;
    MTPHead *h = m->mtp; int H = m->c.hidden;
    for (int r = 0; r < nlogits; r++) {
        int p = pos_base + S - nlogits + r, slot = p & 7;
        memcpy(h->cap + (int64_t)slot * H, x + (int64_t)(S - nlogits + r) * H, (size_t)H * sizeof(float));
        h->cap_pos[slot] = p + 1;
    }
}

static void q36_mtp_settle(Model *m, int ahead_pos, int keep) {
    if (!m->mtp || !m->mtp->loaded) return;
    int want = ahead_pos + keep;
    if (m->mtp->kv_len > want) m->mtp->kv_len = want;   /* stale tail, like the model's */
}

/* One head row at `pos`: own K/V row in, attention over the head's standing
 * rows, y out (also parked in the y-ring for deeper drafts). Mirrors the
 * model's S=1 attention dialect: qk-norm, rope_head_partial, GQA, softmax_row,
 * attn_out *= sigmoid(gate), then o_proj. */
static void q36_mtp_row(Model *m, const float *x, int pos, float *y) {
    MTPHead *h = m->mtp; Cfg *c = &m->c;
    int H = c->hidden, QH = c->q_heads, KVH = c->kv_heads, HD = c->head_dim;
    int QO = QH * c->q_head_dim;                    /* query + gate halves */
    float *q = falloc(QO), *k = falloc((int64_t)KVH * HD), *vv = falloc((int64_t)KVH * HD);
    mtp_gemv(q, x, h->qw, H, QO);
    mtp_gemv(k, x, h->kw, H, KVH * HD);
    mtp_gemv(vv, x, h->vw, H, KVH * HD);
    for (int kvh = 0; kvh < KVH; kvh++) {
        float *kr = k + (int64_t)kvh * HD, *vr = vv + (int64_t)kvh * HD;
        if (h->n_k) rmsnorm_row(kr, kr, h->n_k, HD, c->eps);
        rope_head_partial(kr, pos, c->rotary_dim, HD, c->theta);
        memcpy(h->hk + ((int64_t)kvh * m->max_t + pos) * HD, kr, (size_t)HD * sizeof(float));
        memcpy(h->hv + ((int64_t)kvh * m->max_t + pos) * HD, vr, (size_t)HD * sizeof(float));
    }
    float *ctx = falloc((int64_t)QH * HD);
    for (int qh = 0; qh < QH; qh++) {
        float *qr = q + (int64_t)qh * HD;
        float *gt = q + (int64_t)(QH * HD + qh) * HD;
        if (h->n_q) rmsnorm_row(qr, qr, h->n_q, HD, c->eps);
        rope_head_partial(qr, pos, c->rotary_dim, HD, c->theta);
        int kvh = qh / (QH / KVH);
        const float *Kl = h->hk + (int64_t)kvh * m->max_t * HD;
        const float *Vl = h->hv + (int64_t)kvh * m->max_t * HD;
        float *sc = falloc(pos + 1);
        for (int t = 0; t <= pos; t++) {
            const float *kr = Kl + (int64_t)t * HD;
            float s = 0;
            for (int dd = 0; dd < HD; dd++) s += qr[dd] * kr[dd];
            sc[t] = s / sqrtf((float)HD);
        }
        softmax_row(sc, pos + 1);
        float *cx = ctx + (int64_t)qh * HD;
        for (int dd = 0; dd < HD; dd++) {
            float a = 0;
            for (int t = 0; t <= pos; t++) a += sc[t] * Vl[(int64_t)t * HD + dd];
            cx[dd] = a;
        }
        if (c->attn_output_gate)
            for (int dd = 0; dd < HD; dd++) cx[dd] *= 1.0f / (1.0f + expf(-gt[dd]));
        free(sc);
    }
    mtp_gemv(y, ctx, h->ow, QH * HD, H);
    memcpy(h->yring + (int64_t)pos * H, y, (size_t)H * sizeof(float));
    if (pos + 1 > h->kv_len) h->kv_len = pos + 1;
    free(q); free(k); free(vv); free(ctx);
}

/* Drafts for positions pos+1..: d1 from (embed(tok), model hidden at pos-1),
 * deeper drafts from the head's own previous hidden (qwen38 stream semantics).
 * Q36_MTP_FORCE reroutes to the lookup oracle (verify/rollback test path). */
static int q36_mtp_propose(Model *m, Q36Spec *sp, int tok, int pos, int room, int *d) {
    MTPHead *h = m->mtp; Cfg *c = &m->c; int H = c->hidden, V = c->vocab;
    if (!h->hk) {
        if (m->max_t <= 0) return 0;
        int KVH = c->kv_heads, HD = c->head_dim;
        h->hk = falloc((int64_t)KVH * m->max_t * HD);
        h->hv = falloc((int64_t)KVH * m->max_t * HD);
        h->yring = falloc((int64_t)m->max_t * H);
    }
    if (g_q36_mtp_drafts > 0 && room < g_q36_mtp_drafts) room = room;
    if (g_q36_mtp_drafts > 0 && room > g_q36_mtp_drafts) room = g_q36_mtp_drafts;
    if (room > Q36_SPEC_ROWS - 1) room = Q36_SPEC_ROWS - 1;
    if (room <= 0) return 0;
    if (g_q36_mtp_force[0] && sp) {
        int n = 0;
        while (n < room) { int t = q36_spec_forced(sp, room, n + 1, pos, V); if (t < 0) break; d[n++] = t; }
        return n;
    }
    const float *hid = q36_mtp_hidden_at(m, pos - 1);
    if (!hid) return 0;
    float *e = falloc(H), *cat = falloc(2 * (int64_t)H), *xx = falloc(H);
    float *a = falloc(H), *b = falloc(H);
    q36_embed_row(m, tok, pos, e);
    rmsnorm_row(a, e, h->n_emb, H, c->eps);
    rmsnorm_row(b, hid, h->n_hid, H, c->eps);
    memcpy(cat, a, (size_t)H * sizeof(float)); memcpy(cat + H, b, (size_t)H * sizeof(float));
    mtp_gemv(xx, cat, h->fc, 2 * H, H);
    int n = 0, rowpos = pos;
    while (n < room) {
        float *yy = falloc(H);
        q36_mtp_row(m, xx, rowpos, yy);
        float *nr = falloc(H), *lg = falloc(V);
        rmsnorm_row(nr, yy, h->n_fin, H, c->eps);
        matmul_d(lg, nr, &m->lm_head, 1, H, V);
        int best = 0;
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        d[n++] = best;
        free(nr); free(lg);
        if (n >= room) { free(yy); break; }
        float *e2 = falloc(H);
        q36_embed_row(m, best, rowpos + 1, e2);
        rmsnorm_row(a, e2, h->n_emb, H, c->eps);
        rmsnorm_row(b, yy, h->n_hid, H, c->eps);
        memcpy(cat, a, (size_t)H * sizeof(float)); memcpy(cat + H, b, (size_t)H * sizeof(float));
        mtp_gemv(xx, cat, h->fc, 2 * H, H);
        free(e2); free(yy);
        rowpos++;
    }
    free(e); free(cat); free(xx); free(a); free(b);
    return n;
}

static void q36_spec_begin(Model *m, Q36Spec *sp, const int *prompt, int np) {
""")

# 3. hidden capture at the lm_head seam (+ file-scope fwd decl: step_ex calls it
#    before the definition that lives with the spec machinery)
edit("""static float *step_ex(Model *m, const int *ids, int S, int pos_base, int nlogits) {
""",
     """static void q36_mtp_capture(Model *m, const float *x, int S, int pos_base, int nlogits);
static float *step_ex(Model *m, const int *ids, int S, int pos_base, int nlogits) {
""")

edit("""    for (int r = 0; r < nlogits; r++) {
        float *lr = logit + (int64_t)r * c->vocab;
        rmsnorm_row(last, x + (int64_t)(S - nlogits + r)*D, m->final_norm, D, c->eps);
""",
     """    q36_mtp_capture(m, x, S, pos_base, nlogits);
    for (int r = 0; r < nlogits; r++) {
        float *lr = logit + (int64_t)r * c->vocab;
        rmsnorm_row(last, x + (int64_t)(S - nlogits + r)*D, m->final_norm, D, c->eps);
""")

# 4. Q36Spec: pending-verify source + MTP counters
edit("""    int ended_ahead;             /* the generation ended with a verify's rows unconsumed (undone) */
""",
     """    int cur_src;                 /* source of the pending verify's drafts (SPEC_SRC_*) */
    uint64_t m_drafts, m_accepted, m_verifies;   /* MTP-source counters for the report */
    int ended_ahead;             /* the generation ended with a verify's rows unconsumed (undone) */
""")

# 5. proposal: MTP beside lookup, gate values decide
edit("""    int d[Q36_SPEC_ROWS], k = 0;
    if (sp->lookup && room > 0) {
        int n = 0;
        if (sp->force)
            while (n < room) { int t = q36_spec_forced(sp, room, n + 1, pos, V); if (t < 0) break; d[n++] = t; }
        else n = spec_lookup(sp->hist, sp->hist_n, 2, 4, room, d);
        if (n > 0) k = spec_gate_pick(&sp->gate, SPEC_SRC_LOOKUP, n, NULL);
        if (k > 0 && !q36_spec_alloc(m, k)) k = 0;
    }
""",
     """    int d[Q36_SPEC_ROWS], k = 0;
    int src = SPEC_SRC_LOOKUP;
    if (m->mtp && m->mtp->loaded && room > 0) {
        int dm[Q36_SPEC_ROWS], dl[Q36_SPEC_ROWS], nm = 0, nl = 0;
        nm = q36_mtp_propose(m, sp, tok, pos, room, dm);
        if (sp->lookup) {
            if (sp->force)
                while (nl < room) { int t = q36_spec_forced(sp, room, nl + 1, pos, V); if (t < 0) break; dl[nl++] = t; }
            else nl = spec_lookup(sp->hist, sp->hist_n, 2, 4, room, dl);
        }
        double vm = 0, vl = 0;
        int km = nm > 0 ? spec_gate_pick(&sp->gate, SPEC_SRC_MTP, nm, &vm) : 0;
        int kl = nl > 0 ? spec_gate_pick(&sp->gate, SPEC_SRC_LOOKUP, nl, &vl) : 0;
        if (km > 0 && vm >= vl)      { memcpy(d, dm, (size_t)km * sizeof(int)); k = km; src = SPEC_SRC_MTP; }
        else if (kl > 0)             { memcpy(d, dl, (size_t)kl * sizeof(int)); k = kl; src = SPEC_SRC_LOOKUP; }
        else if (km > 0)             { memcpy(d, dm, (size_t)km * sizeof(int)); k = km; src = SPEC_SRC_MTP; }
        if (k > 0 && !q36_spec_alloc(m, k)) k = 0;
    } else if (sp->lookup && room > 0) {
        int n = 0;
        if (sp->force)
            while (n < room) { int t = q36_spec_forced(sp, room, n + 1, pos, V); if (t < 0) break; d[n++] = t; }
        else n = spec_lookup(sp->hist, sp->hist_n, 2, 4, room, d);
        if (n > 0) k = spec_gate_pick(&sp->gate, SPEC_SRC_LOOKUP, n, NULL);
        if (k > 0 && !q36_spec_alloc(m, k)) k = 0;
    }
    sp->cur_src = src;
""")

# 6. counters at the verify
edit("""    sp->forwards++; sp->drafts += (uint64_t)k; sp->verifies++;
""",
     """    sp->forwards++; sp->drafts += (uint64_t)k; sp->verifies++;
    if (src == SPEC_SRC_MTP) { sp->m_drafts += (uint64_t)k; sp->m_verifies++; }
""")

# 7. settle: head-row rollback + per-source gate accounting
edit("""    if (keep < sp->ahead_n) q36_spec_rollback(m, sp->ahead_pos + keep, keep);
    spec_gate_result(&sp->gate, SPEC_SRC_LOOKUP, keep - 1 + (rejected ? 1 : 0), keep - 1);
""",
     """    if (keep < sp->ahead_n) q36_spec_rollback(m, sp->ahead_pos + keep, keep);
    q36_mtp_settle(m, sp->ahead_pos, keep);
    spec_gate_result(&sp->gate, sp->cur_src, keep - 1 + (rejected ? 1 : 0), keep - 1);
""")

# 8. accepted counter per source
edit("""            sp->accepted++;
""",
     """            sp->accepted++;
            if (sp->cur_src == SPEC_SRC_MTP) sp->m_accepted++;
""")

# 9. report line
edit("""            sp->gate.off ? "off" : "on", (unsigned long long)sp->gate.declined[SPEC_SRC_LOOKUP],
            (unsigned long long)sp->gate.probes[SPEC_SRC_LOOKUP], sp->ended_ahead ? " | ended mid-verify" : "");
""",
     """            sp->gate.off ? "off" : "on", (unsigned long long)sp->gate.declined[SPEC_SRC_LOOKUP],
            (unsigned long long)sp->gate.probes[SPEC_SRC_LOOKUP], sp->ended_ahead ? " | ended mid-verify" : "");
    if (sp->m_verifies || sp->m_drafts) {
        char g[384];
        spec_gate_describe(&sp->gate, SPEC_SRC_MTP, g, sizeof g);
        fprintf(stderr, "[qwen36 MTP] %s: acceptance %.1f%% (%llu/%llu drafts in %llu verifies) | gate %s%s\\n",
                scope, sp->m_drafts ? 100.0 * sp->m_accepted / sp->m_drafts : 0.0,
                (unsigned long long)sp->m_accepted, (unsigned long long)sp->m_drafts,
                (unsigned long long)sp->m_verifies, g, sp->ended_ahead ? " | ended mid-verify" : "");
    }
""")

# 10. capture ring allocation at attach
edit("""    h->loaded = 1;
    m->mtp = h;
""",
     """    h->cap = falloc(8 * (int64_t)m->c.hidden);
    for (int i = 0; i < 8; i++) h->cap_pos[i] = 0;
    h->loaded = 1;
    m->mtp = h;
""")

# 11. free the new pools
edit("""    free(h->gw); free(h->uw); free(h->dw);
    free(h); m->mtp = NULL;
""",
     """    free(h->gw); free(h->uw); free(h->dw);
    free(h->hk); free(h->hv); free(h->yring); free(h->cap);
    free(h); m->mtp = NULL;
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_mtp_draft: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".predraft"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mtp_draft applied (%d edits, backup %s)" % (len(EDITS), bak))
