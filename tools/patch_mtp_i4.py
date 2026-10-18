#!/usr/bin/env python3
"""patch_mtp_i4.py — Phase 2b part 5a: RESIDENT int4 head experts (dequant-on-read).

3b unpacked head experts to f32 at attach (2.4 GB on the 128-expert synth) —
fine for CI, impossible for a real 35B-class head inside the 8 GB envelope.
Now the merged int4-gs packed bytes + per-group scales stay resident exactly
as stored in model-mtp.safetensors (design: "the head's experts bypass the
LRU store"), and the head MLP dequantizes on read:

  mtp_gemv_i4gs: per output row, dequant the row's nibbles into a scratch and
  dot with x (rows are independent => OpenMP-safe, deterministic)

Also drops the now-dead mtp_unpack_i4_gs expansion helper.

Gates: unchanged CI MoE gates (same math, same bits? dequant-on-read computes
the same products as the unpacked path, so sequences stay identical to the
previous green run) + RAM: attach no longer scales with expert count x f32.

Anchored exact-byte edits, house style. Backup qwen36.c.prei4.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "mtp_gemv_i4gs" in src:
    raise SystemExit("patch_mtp_i4: already applied")
if "q36_mtp_mlp" not in src:
    raise SystemExit("patch_mtp_i4: part 3b (patch_mtp_moe.py) must be applied first")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# 1. struct: packed residents instead of unpacked f32
edit("""    float **eg, **eu, **ed;      /* per-expert unpacked f32 (gate/up [I,H], down [H,I]) */
""",
     """    uint8_t **eq;                /* per-expert merged int4-gs packed bytes, resident */
    float **es;                  /* per-expert merged per-group scales, resident */
""")

# 2. drop the dead expansion helper
edit("""/* int4-gs packed (nibble lo|hi, per-group scale) -> resident f32 rows */
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

""", "")

# 3. dequant-on-read GEMV, beside mtp_gemv
edit("""static const float *q36_mtp_hidden_at(Model *m, int p) {
""",
     """/* y[o] = sum_i x[i] * dequant(pq, qs)[qoff + o*In + i]; rows independent, so
 * the per-row scratch makes this OpenMP-safe and bit-deterministic. qoff must
 * be even (every merged part has an even element count). */
static void mtp_gemv_i4gs(float *y, const float *x, const uint8_t *pq, const float *qs,
                          int64_t qoff, int64_t soff, int O, int In, int gs) {
    int ng = In / gs;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int o = 0; o < O; o++) {
        float *t = malloc((size_t)In * sizeof(float));
        if (!t) { fprintf(stderr, "[qwen36 MTP] OOM row scratch\\n"); exit(1); }
        const uint8_t *rb = pq + (qoff + (int64_t)o * In) / 2;
        const float *rs = qs + soff + (int64_t)o * ng;
        for (int i = 0; i < In; i++) {
            int b = rb[i / 2];
            int nib = (i & 1) ? (b >> 4) & 0xF : b & 0xF;
            int qv = nib > 7 ? nib - 16 : nib;
            t[i] = (float)qv * rs[i / gs];
        }
        float s = 0;
        for (int i = 0; i < In; i++) s += x[i] * t[i];
        y[o] = s;
        free(t);
    }
}

static const float *q36_mtp_hidden_at(Model *m, int p) {
""")

# 4. attach: store packed bytes + scales, no expansion
edit("""            h->eg = calloc((size_t)h->E, sizeof(float *));
            h->eu = calloc((size_t)h->E, sizeof(float *));
            h->ed = calloc((size_t)h->E, sizeof(float *));
            if (!h->eg || !h->eu || !h->ed) { fprintf(stderr, "[qwen36 MTP] OOM experts\\n"); exit(1); }
""",
     """            h->eq = calloc((size_t)h->E, sizeof(uint8_t *));
            h->es = calloc((size_t)h->E, sizeof(float *));
            if (!h->eq || !h->es) { fprintf(stderr, "[qwen36 MTP] OOM experts\\n"); exit(1); }
""")
edit("""                st_tensor *tq = st_find(&m->S, qn), *ts = st_find(&m->S, sn);
                if (!tq || !ts) { fprintf(stderr, "[qwen36 MTP] missing %s\\n", qn); exit(1); }
                int64_t nel = 2 * gu + du;
                uint8_t *pq = malloc((size_t)(nel / 2));
                int64_t nsc = 2 * (gu / h->GS) + (du / h->GS);
                float *qs = malloc((size_t)nsc * sizeof(float));
                if (!pq || !qs) { fprintf(stderr, "[qwen36 MTP] OOM %s\\n", qn); exit(1); }
                st_read_raw(&m->S, qn, pq, 1);
                if (st_read_f32(&m->S, sn, qs, 0) != nsc) { fprintf(stderr, "[qwen36 MTP] bad scales %s\\n", sn); exit(1); }
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
""",
     """                st_tensor *tq = st_find(&m->S, qn), *ts = st_find(&m->S, sn);
                if (!tq || !ts) { fprintf(stderr, "[qwen36 MTP] missing %s\\n", qn); exit(1); }
                int64_t nel = 2 * gu + du;
                int64_t nsc = 2 * (gu / h->GS) + (du / h->GS);
                if (nel & 1) { fprintf(stderr, "[qwen36 MTP] odd merged numel %s\\n", qn); exit(1); }
                h->eq[e] = malloc((size_t)(nel / 2));
                h->es[e] = malloc((size_t)nsc * sizeof(float));
                if (!h->eq[e] || !h->es[e]) { fprintf(stderr, "[qwen36 MTP] OOM %s\\n", qn); exit(1); }
                st_read_raw(&m->S, qn, h->eq[e], 1);
                if (st_read_f32(&m->S, sn, h->es[e], 0) != nsc) { fprintf(stderr, "[qwen36 MTP] bad scales %s\\n", sn); exit(1); }
""")

# 5. head MLP: dequant-on-read per part
edit("""    float *t = falloc(I), *gg = falloc(I), *uu = falloc(I), *tmp = falloc(H);
    for (int kk = 0; kk < K; kk++) {
        int e = idx[kk];
        float w = wsum > 0 ? lg[idx[kk]] / wsum : 0;
        mtp_gemv(gg, x, h->eg[e], H, I);
        mtp_gemv(uu, x, h->eu[e], H, I);
        for (int i = 0; i < I; i++) t[i] = (gg[i] / (1.0f + expf(-gg[i]))) * uu[i];
        mtp_gemv(tmp, t, h->ed[e], I, H);
        for (int d = 0; d < H; d++) y[d] += w * tmp[d];
    }
""",
     """    float *t = falloc(I), *gg = falloc(I), *uu = falloc(I), *tmp = falloc(H);
    int64_t gu = (int64_t)I * H;                 /* merged layout: gate | up | down */
    for (int kk = 0; kk < K; kk++) {
        int e = idx[kk];
        float w = wsum > 0 ? lg[idx[kk]] / wsum : 0;
        mtp_gemv_i4gs(gg, x, h->eq[e], h->es[e], 0, 0, I, H, h->GS);
        mtp_gemv_i4gs(uu, x, h->eq[e], h->es[e], gu, gu / h->GS, I, H, h->GS);
        for (int i = 0; i < I; i++) t[i] = (gg[i] / (1.0f + expf(-gg[i]))) * uu[i];
        mtp_gemv_i4gs(tmp, t, h->eq[e], h->es[e], 2 * gu, 2 * (gu / h->GS), H, I, h->GS);
        for (int d = 0; d < H; d++) y[d] += w * tmp[d];
    }
""")

# 6. free the packed residents
edit("""    if (h->eg) for (int e = 0; e < h->E; e++) { free(h->eg[e]); free(h->eu[e]); free(h->ed[e]); }
    free(h->eg); free(h->eu); free(h->ed);
""",
     """    if (h->eq) for (int e = 0; e < h->E; e++) { free(h->eq[e]); free(h->es[e]); }
    free(h->eq); free(h->es);
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_mtp_i4: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".prei4"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mtp_i4 applied (%d edits, backup %s)" % (len(EDITS), bak))
