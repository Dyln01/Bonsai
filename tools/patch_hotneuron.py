#!/usr/bin/env python3
"""patch_hotneuron.py — HOT-NEURON PAGING for dense models (TRUNK_MAP v3). CLEAN."""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
if "HOT_FRAC" in src:
    print("already patched"); sys.exit(0)

def sub1(anchor, repl, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag}: count={src.count(anchor)}"
    src = src.replace(anchor, repl)

# 1) env knob
sub1("static int g_q36_ref_mode;",
"""static int g_q36_ref_mode;
static double q36_hot_frac(void) {
    static double f = -1;
    if (f < 0) {
        const char *e = getenv("HOT_FRAC");
        f = (e && *e) ? atof(e) : 1.0;
        if (!(f > 0.0 && f <= 1.0)) { fprintf(stderr, "HOT_FRAC must be in (0,1]\\n"); exit(1); }
    }
    return f;
}""", "hotfrac")

# 2) paged shared-expert path
A_SH = """static void qwen_shared_experts_cpu(Model *m, Layer *l, const float *x, int S,
                                    float *out, float *g, float *u, float *hh) {
    Cfg *c=&m->c; int D=c->hidden, I=c->shared_inter;
    if (I <= 0) return;   /* no shared expert (qwen3_moe) */
    int B=qwen_shared_batch_rows(S,D,I);"""
sub1(A_SH,
"""static void q36_down_prefix(const QW *dw, float *dst, const float *g, int D, int I, int Ik) {
    /* dst[d] = sum_{i<Ik} down[d][i] * g[i], for hotness-sorted containers
     * (kept neurons are elements 0..Ik of every row). Three storage formats. */
    if (dw->q) {
        const int8_t *q = dw->q; const float *sc = dw->sc;
        #pragma omp parallel for schedule(static)
        for (int d = 0; d < D; d++) {
            const int8_t *row = q + (int64_t)d * I;
            float acc = 0;
            for (int i = 0; i < Ik; i++) acc += (float)row[i] * g[i];
            dst[d] = acc * (sc ? sc[d] : 1.0f);
        }
    } else if (dw->map && dw->map_len == (int64_t)D*I/2) {
        const uint8_t *mp = (const uint8_t*)dw->map; const float *sc = dw->sc;
        #pragma omp parallel for schedule(static)
        for (int d = 0; d < D; d++) {
            const uint8_t *row = mp + (int64_t)d * (I >> 1);
            float acc = 0;
            for (int i = 0; i < Ik; i++) {
                uint8_t b = row[i >> 1];
                int8_t e = (int8_t)((i & 1) ? (b >> 4) : (b & 0xF));
                if (e & 8) e -= 16;
                acc += (float)e * g[i];
            }
            dst[d] = acc * (sc ? sc[d] : 1.0f);
        }
    } else if (dw->map) {  /* f16 mapped */
        const uint16_t *mp = (const uint16_t*)dw->map;
        #pragma omp parallel for schedule(static)
        for (int d = 0; d < D; d++) {
            const uint16_t *row = mp + (int64_t)d * I;
            float acc = 0;
            for (int i = 0; i < Ik; i++) {
                uint16_t h = row[i];
                uint32_t bits = ((uint32_t)(h & 0x8000u) << 16) | ((uint32_t)((h >> 10) & 0x1Fu) << 23) | ((uint32_t)(h & 0x3FFu) << 13);
                if (((h >> 10) & 0x1F) == 0) bits = (uint32_t)(h & 0x8000u) << 16;
                float wv; memcpy(&wv, &bits, 4);
                acc += wv * g[i];
            }
            dst[d] = acc;
        }
    } else if (dw->w) {
        const float *w = dw->w;
        #pragma omp parallel for schedule(static)
        for (int d = 0; d < D; d++) {
            const float *row = w + (int64_t)d * I;
            float acc = 0;
            for (int i = 0; i < Ik; i++) acc += row[i] * g[i];
            dst[d] = acc;
        }
    }
}

static void qwen_shared_experts_cpu(Model *m, Layer *l, const float *x, int S,
                                    float *out, float *g, float *u, float *hh) {
    Cfg *c=&m->c; int D=c->hidden, I=c->shared_inter;
    if (I <= 0) return;   /* no shared expert (qwen3_moe) */
    /* HOT-NEURON PAGING (TRUNK_MAP v3, HOT_FRAC<1): containers are
     * hotness-sorted, so the top-HOT_FRAC neurons are rows/cols [0,Ik).
     * gate/up: narrowed QW (O=Ik) through the normal kernels; down: prefix
     * columns via q36_down_prefix. Tail never touched. */
    double hk = q36_hot_frac();
    int Ik = (int)(I * hk);
    if (Ik < 1) Ik = 1;
    if (Ik < I) {
        QW ng = l->sh_g, nu = l->sh_u;
        ng.O = Ik; ng.I = D; ng.q4 = NULL; ng.sg = NULL; ng.ng = 0;
        nu.O = Ik; nu.I = D; nu.q4 = NULL; nu.sg = NULL; nu.ng = 0;
        if (l->sh_g.map) { ng.map_len = (l->sh_g.map_len == (int64_t)D*I/2) ? (int64_t)D*Ik/2 : l->sh_g.map_len; ng.sc = l->sh_g.sc; }
        if (l->sh_u.map) { nu.map_len = (l->sh_u.map_len == (int64_t)D*I/2) ? (int64_t)D*Ik/2 : l->sh_u.map_len; nu.sc = l->sh_u.sc; }
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*D;
            matmul_d(g, xs, &ng, 1, D, Ik);
            matmul_d(u, xs, &nu, 1, D, Ik);
            for (int i = 0; i < Ik; i++) { float sv = g[i]; g[i] = (sv/(1.f+expf(-sv)))*u[i]; }
            q36_down_prefix(&l->sh_d, hh, g, D, I, Ik);
            float sgate = 1.f;
            if (l->sh_gate) { float sg = 0.f; for (int i = 0; i < D; i++) sg += xs[i]*l->sh_gate[i]; sgate = 1.f/(1.f+expf(-sg)); }
            float *os = out + (int64_t)s*D;
            for (int d = 0; d < D; d++) os[d] += sgate * hh[d];
        }
        return;
    }
    int B=qwen_shared_batch_rows(S,D,I);""", "shared-paged")

open(P, "w").write(src)
print("HOT-NEURON PAGING (v3) applied cleanly")
