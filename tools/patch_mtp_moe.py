#!/usr/bin/env python3
"""patch_mtp_moe.py — Phase 2b part 3b: MoE head + the missing MLP stage.

  - head layer fidelity fix: q36_mtp_row ended at o_proj — the post-attention
    norm + MLP stage was missing (draft quality only, gates were unaffected,
    but the head math must match the contract). Now every head row runs
    rmsnorm(post_attention_layernorm) + MLP before y is parked.
  - dense MLP: silu(gate)*up -> down (the mlp. prefix weights from part 1)
  - MoE MLP: router softmax -> group-limited top-k (n_group/topk_group from the
    container config, mirroring moe_ex) -> renormalised weights -> per-expert
    silu FFN from UNPACKED resident f32 (int4-gs64 merged_weight decoded at
    attach; resident-int4 dequant-on-read is follow-up 3c) + optional shared
    expert with its sigmoid gate
  - attach no longer refuses MoE heads; report line shows dense=0

Gates (CI): coder30 MoE synth + its MTP head: load line dense=0; drafting-on
id-sequence == stock; forced-accept sequence == stock + acceptance 100%.

Anchored exact-byte edits, house style. Backup qwen36.c.premoe.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "q36_mtp_mlp" in src:
    raise SystemExit("patch_mtp_moe: already applied")
if "q36_mtp_propose" not in src:
    raise SystemExit("patch_mtp_moe: part 2 must be applied first")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# 1. MoE fields on the head
edit("""    float *cap; int cap_pos[8]; /* captured model hidden, ring of 8 positions */
} MTPHead;
""",
     """    float *cap; int cap_pos[8];
    int E, SI, GS;               /* MoE head: experts, shared inter, expert group size */
    float *rw;                   /* router [E,H] f32 resident */
    float **eg, **eu, **ed;      /* per-expert unpacked f32 (gate/up [I,H], down [H,I]) */
    float *sg, *su, *sd, *shg;   /* shared expert + its scalar gate */
} MTPHead;
""")

# 2. unpack helper + head MLP (dense + MoE), before the head row function
edit("""static void q36_mtp_row(Model *m, const float *x, int pos, float *y) {
""",
     """/* int4-gs packed (nibble lo|hi, per-group scale) -> resident f32 rows */
static float *mtp_unpack_i4_gs(shards *S, const char *qn, const char *sn, int O, int In, int gs) {
    st_tensor *tq = st_find(S, qn), *ts = st_find(S, sn);
    if (!tq || !ts) { fprintf(stderr, "[qwen36 MTP] missing %s/%s\\n", qn, sn); exit(1); }
    int64_t nel = (int64_t)O * In;
    uint8_t *pq = malloc((size_t)(nel / 2));
    float *qs = malloc((size_t)(nel / gs) * sizeof(float));
    if (!pq || !qs) { fprintf(stderr, "[qwen36 MTP] OOM unpack %s\\n", qn); exit(1); }
    st_read_raw(S, qn, pq, 1);
    if (st_read_f32(S, sn, qs, 0) != nel / gs) { fprintf(stderr, "[qwen36 MTP] bad scale numel %s\\n", sn); exit(1); }
    float *w = falloc(nel);
    int ng = In / gs;
    for (int64_t i = 0; i < nel; i++) {
        int b = pq[i / 2];
        int nib = (i & 1) ? (b >> 4) & 0xF : b & 0xF;
        int qv = nib > 7 ? nib - 16 : nib;
        int64_t o = i / In;
        w[i] = (float)qv * qs[o * ng + (i - o * In) / gs];
    }
    free(pq); free(qs);
    return w;
}

/* head MLP: dense (mlp. prefix) or MoE (router -> group-limited top-k ->
 * renormalised expert mix + optional sigmoid-gated shared expert), mirroring
 * moe_ex's routing dialect on the container's own config. */
static void q36_mtp_mlp(Model *m, const float *x, float *y) {
    MTPHead *h = m->mtp; Cfg *c = &m->c; int H = c->hidden;
    memset(y, 0, (size_t)H * sizeof(float));
    if (h->dense) {
        int I = h->I;
        float *g = falloc(I), *u = falloc(I), *t = falloc(I);
        mtp_gemv(g, x, h->gw, H, I);
        mtp_gemv(u, x, h->uw, H, I);
        for (int i = 0; i < I; i++) t[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
        mtp_gemv(y, t, h->dw, I, H);
        free(g); free(u); free(t);
        return;
    }
    int E = h->E, I = h->I, K = c->topk;
    float *lg = falloc(E);
    mtp_gemv(lg, x, h->rw, H, E);
    softmax_row(lg, E);
    int idx[64]; float wv[64];
    if (K > 64) K = 64;
    if (c->n_group > 0) {                     /* group-limited top-k, as moe_ex */
        int G = c->n_group, ES = E / G;
        float gsc[64];
        if (G > 64) { fprintf(stderr, "[qwen36 MTP] n_group > 64 unsupported\\n"); exit(1); }
        for (int g = 0; g < G; g++) {
            float b1 = -1, b2 = -1;
            for (int e = 0; e < ES; e++) {
                float v = lg[g * ES + e];
                if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
            }
            gsc[g] = b1 + (b2 > 0 ? b2 : 0);
        }
        uint8_t gkeep[64] = {0};
        for (int kk = 0; kk < c->topk_group; kk++) {
            int bg = -1; float bv = -1;
            for (int g = 0; g < G; g++) if (!gkeep[g] && gsc[g] > bv) { bv = gsc[g]; bg = g; }
            if (bg < 0) break;
            gkeep[bg] = 1;
        }
        int n = 0;
        for (int kk = 0; kk < K; kk++) {
            int be = -1; float bv = -1;
            for (int g = 0; g < G; g++) if (gkeep[g])
                for (int e = 0; e < ES; e++) {
                    int cand = g * ES + e, taken = 0;
                    for (int j = 0; j < n; j++) if (idx[j] == cand) taken = 1;
                    if (!taken && lg[cand] > bv) { bv = lg[cand]; be = cand; }
                }
            if (be < 0) break;
            idx[n++] = be;
        }
        K = n;
    } else {
        for (int kk = 0; kk < K; kk++) {
            int be = -1; float bv = -1;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int j = 0; j < kk; j++) if (idx[j] == e) taken = 1;
                if (!taken && lg[e] > bv) { bv = lg[e]; be = e; }
            }
            if (be < 0) break;
            idx[kk] = be; wv[kk] = bv;
        }
    }
    float wsum = 0;
    for (int kk = 0; kk < K; kk++) wsum += lg[idx[kk]];
    float *t = falloc(I), *gg = falloc(I), *uu = falloc(I), *tmp = falloc(H);
    for (int kk = 0; kk < K; kk++) {
        int e = idx[kk];
        float w = wsum > 0 ? lg[idx[kk]] / wsum : 0;
        mtp_gemv(gg, x, h->eg[e], H, I);
        mtp_gemv(uu, x, h->eu[e], H, I);
        for (int i = 0; i < I; i++) t[i] = (gg[i] / (1.0f + expf(-gg[i]))) * uu[i];
        mtp_gemv(tmp, t, h->ed[e], I, H);
        for (int d = 0; d < H; d++) y[d] += w * tmp[d];
    }
    if (h->SI) {
        float sgt = 0;
        for (int d = 0; d < H; d++) sgt += x[d] * h->shg[d];
        sgt = 1.0f / (1.0f + expf(-sgt));
        mtp_gemv(gg, x, h->sg, H, h->SI);
        mtp_gemv(uu, x, h->su, H, h->SI);
        for (int i = 0; i < h->SI; i++) t[i] = (gg[i] / (1.0f + expf(-gg[i]))) * uu[i];
        mtp_gemv(tmp, t, h->sd, h->SI, H);
        for (int d = 0; d < H; d++) y[d] += sgt * tmp[d];
    }
    free(lg); free(t); free(gg); free(uu); free(tmp);
}

static void q36_mtp_row(Model *m, const float *x, int pos, float *y) {
""")

# 3. the missing MLP stage in the head row
edit("""    mtp_gemv(y, ctx, h->ow, QH * HD, H);
    memcpy(h->yring + (int64_t)pos * H, y, (size_t)H * sizeof(float));
""",
     """    mtp_gemv(y, ctx, h->ow, QH * HD, H);
    {   /* post-attention norm + MLP: the head layer is attn THEN mlp (contract) */
        float *pn = falloc(H), *my = falloc(H);
        rmsnorm_row(pn, y, h->n_post, H, c->eps);
        q36_mtp_mlp(m, pn, my);
        memcpy(y, my, (size_t)H * sizeof(float));
        free(pn); free(my);
    }
    memcpy(h->yring + (int64_t)pos * H, y, (size_t)H * sizeof(float));
""")

# 4. attach: load the MoE head instead of refusing
edit("""        } else {
            fprintf(stderr, "[qwen36 MTP] MoE head in container; part 1 loads dense heads only (part 2 adds experts) — refusing\\n");
            exit(1);
        }
""",
     """        } else if (st_find(&m->S, "mtp.layers.0.mlp.gate.weight")) {
            h->dense = 0;
            h->E = m->c.n_experts; h->I = m->c.inter; h->SI = m->c.shared_inter;
            h->GS = m->c.expert_gs > 0 ? m->c.expert_gs : 64;
            int HH = m->c.hidden;
            if (h->E <= 0 || h->I <= 0 || h->E > 1024) {
                fprintf(stderr, "[qwen36 MTP] head expert config E=%d I=%d out of range — refusing\\n", h->E, h->I);
                exit(1);
            }
            h->rw = mtp_f32(&m->S, "mtp.layers.0.mlp.gate.weight", NULL);
            h->eg = calloc((size_t)h->E, sizeof(float *));
            h->eu = calloc((size_t)h->E, sizeof(float *));
            h->ed = calloc((size_t)h->E, sizeof(float *));
            if (!h->eg || !h->eu || !h->ed) { fprintf(stderr, "[qwen36 MTP] OOM experts\\n"); exit(1); }
            for (int e = 0; e < h->E; e++) {
                char qn[256], sn[256];
                snprintf(qn, sizeof qn, "mtp.layers.0.mlp.experts.%d.merged_weight", e);
                snprintf(sn, sizeof sn, "mtp.layers.0.mlp.experts.%d.qs", e);
                int64_t gu = (int64_t)h->I * HH, du = (int64_t)HH * h->I;
                float *all = mtp_unpack_i4_gs(&m->S, qn, sn, 0, 0, 0);   /* placeholder, replaced below */
                free(all);
                /* merged layout: [gate IxH | up IxH | down HxI], each int4-gs packed;
                 * unpack the three parts with their own row/col geometry */
                st_tensor *tq = st_find(&m->S, qn), *ts = st_find(&m->S, sn);
                if (!tq || !ts) { fprintf(stderr, "[qwen36 MTP] missing %s\\n", qn); exit(1); }
                int64_t nel = 2 * gu + du;
                uint8_t *pq = malloc((size_t)(nel / 2));
                int64_t nsc = 2 * (gu / h->GS) + (du / h->GS);
                float *qs = malloc((size_t)nsc * sizeof(float));
                if (!pq || !qs) { fprintf(stderr, "[qwen36 MTP] OOM %s\\n", qn); exit(1); }
                st_read_raw(&m->S, qn, pq, 1);
                if (st_read_f32(&m->S, sn, qs, 0) != nsc) { fprintf(stderr, "[qwen36 MTP] bad scales %s\\n", sn); exit(1); }
                auto_unpack: ;
                const int64_t parts[3][2] = { { h->I, HH }, { h->I, HH }, { HH, h->I } };
                int64_t qoff = 0, soff = 0;
                float **dst[3] = { &h->eg[e], &h->eu[e], &h->ed[e] };
                for (int pi = 0; pi < 3; pi++) {
                    int O = (int)parts[pi][0], In = (int)parts[pi][1];
                    int ng = In / h->GS;
                    float *w = falloc((int64_t)O * In);
                    for (int64_t i = 0; i < (int64_t)O * In; i++) {
                        int b = pq[qoff / 2 + i / 2];
                        int nib = (i & 1) ? (b >> 4) & 0xF : b & 0xF;
                        int qv = nib > 7 ? nib - 16 : nib;
                        int64_t o = i / In;
                        w[i] = (float)qv * qs[soff + o * ng + (i - o * In) / h->GS];
                    }
                    *dst[pi] = w;
                    qoff += (int64_t)O * In;
                    soff += (int64_t)O * ng;
                }
                free(pq); free(qs);
            }
            if (h->SI > 0) {
                h->sg  = mtp_f32(&m->S, "mtp.layers.0.mlp.shared_expert.gate_proj.weight", NULL);
                h->su  = mtp_f32(&m->S, "mtp.layers.0.mlp.shared_expert.up_proj.weight", NULL);
                h->sd  = mtp_f32(&m->S, "mtp.layers.0.mlp.shared_expert.down_proj.weight", NULL);
                h->shg = mtp_f32(&m->S, "mtp.layers.0.mlp.shared_expert_gate.weight", NULL);
            }
        } else {
            fprintf(stderr, "[qwen36 MTP] neither dense (mlp.) nor MoE (mlp.gate) head tensors found — refusing\\n");
            exit(1);
        }
""")

# 5. free the MoE residents
edit("""    free(h->hk); free(h->hv); free(h->yring); free(h->cap);
""",
     """    free(h->hk); free(h->hv); free(h->yring); free(h->cap);
    if (h->eg) for (int e = 0; e < h->E; e++) { free(h->eg[e]); free(h->eu[e]); free(h->ed[e]); }
    free(h->eg); free(h->eu); free(h->ed);
    free(h->rw); free(h->sg); free(h->su); free(h->sd); free(h->shg);
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_mtp_moe: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

# clean the placeholder hack (kept the edit anchor simple; drop the dead call)
src = src.replace("""                float *all = mtp_unpack_i4_gs(&m->S, qn, sn, 0, 0, 0);   /* placeholder, replaced below */
                free(all);
""", "")
src = src.replace("                auto_unpack: ;\n", "")

bak = P + ".premoe"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mtp_moe applied (%d edits, backup %s)" % (len(EDITS), bak))
