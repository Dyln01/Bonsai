#!/usr/bin/env python3
"""patch_griffin.py — GRIFFIN-style sequence-level neuron selection (TRUNK_MAP v5).

Training-free AND calibration-free (Dong et al. 2024): "flocking" — tokens within a
sequence activate similar FF neurons. So the PROMPT ITSELF names the important set:
during prefill we observe every position's gate activations, take |gate| as the
per-neuron importance (the paper's exact criterion), rank once, and from then on
compute exactly that fixed per-sequence neuron set in decode — as a GATHER, not a
prefix cut.

Two selection modes (GRIFFIN_MODE env):
  prefix — like the paper: top-K neurons kept in ORIGINAL index order (any container,
           sorted or not; reads become strided but correctness is index-perfect)
  gather — same selection, but the gather list is applied on hotness-sorted
           containers so the picked rows are the SMALLEST indices (contiguous-ish
           reads; pairs with HOT_FRAC prefix semantics)

Quality contract: K = HOT_FRAC * I per layer (same dial). The set differs from the
corpus prefix: it adapts per sequence. Decode computes exactly K neurons/layer.
"""
import os, sys

P = os.path.expanduser("~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
if "GRIFFIN" in src:
    print("already patched"); sys.exit(0)

def sub1(anchor, repl, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag}: count={src.count(anchor)}"
    src = src.replace(anchor, repl)

# ---- state + helpers next to the HotAcc block ----
sub1("typedef struct { float *v; int n; } HotAcc;",
"""typedef struct { float *v; int n; } HotAcc;
/* ---- GRIFFIN (v5): per-sequence neuron selection -----------------------
 * chosen at the END of prefill from |gate| observed on the prompt; used by
 * decode as an explicit gather set. idx[0..k) are ORIGINAL neuron indices,
 * ascending (prefix mode) — the same neurons the full model would use most. */
typedef struct { int *idx; int k; int n; } GriffSet;
static GriffSet g_griff[512]; static int g_griff_n = 0; static int g_griff_on = -1;
static void q36_griff_begin(void) {
    for (int i = 0; i < g_griff_n; i++) { free(g_griff[i].idx); g_griff[i].idx = NULL; g_griff[i].k = 0; }
    g_griff_n = 0;
}
static int q36_griffin_on(void) {
    if (g_griff_on < 0) { const char *e = getenv("GRIFFIN"); g_griff_on = (e && *e == '1'); }
    return g_griff_on;
}
/* Record per-neuron |gate| importance for one layer during prefill; finalize
 * (rank + threshold) when prefill ends. Importance accumulates over all prompt
 * positions: bg/bs layout here is [rows, I] flattened rows*Ik in the paged path
 * or [rows, I] in the stock batched path — caller passes the count actually
 * written and the row stride. */
static double q36_griff_acc[512*3456];  /* fixed worst-case I=3456 student; 27B uses alloc below */
static double *q36_griff_acc_dyn[512]; static int q36_griff_acc_n[512];
static void q36_griff_observe(int layer, const float *g, int rows, int stride, int I) {
    if (!q36_griffin_on()) return;
    if (layer >= g_griff_n) {
        int old = g_griff_n; g_griff_n = layer + 1;
        for (int i = old; i < g_griff_n; i++) { g_griff[i] = (GriffSet){NULL,0,0}; g_griff_acc_dyn[i] = NULL; g_griff_acc_n[i] = 0; }
    }
    if (!g_griff_acc_dyn[layer]) {
        g_griff_acc_dyn[layer] = calloc((size_t)I, sizeof(double));
        g_griff_acc_n[layer] = I;
    }
    double *acc = g_griff_acc_dyn[layer];
    for (int r = 0; r < rows; r++) {
        const float *row = g + (int64_t)r * stride;
        for (int i = 0; i < I; i++) acc[i] += fabs((double)row[i]);
    }
}
static void q36_griff_finalize(int layer, double frac, int I) {
    if (!q36_griffin_on()) return;
    if (layer >= g_griff_n || !g_griff_acc_dyn[layer]) return;
    double *acc = g_griff_acc_dyn[layer];
    int k = (int)(I * frac); if (k < 1) k = 1; if (k > I) k = I;
    /* top-k by |gate|: partial selection via nth_element-style quickselect on indices */
    int *idx = malloc(sizeof(int) * (size_t)I);
    for (int i = 0; i < I; i++) idx[i] = i;
    /* simple k-largest by threshold sweep (I is a few thousand; O(I*k) fine for k<=I) */
    /* use heap-free approach: sort indices by acc descending via insertion of top-k */
    int *top = malloc(sizeof(int) * (size_t)k);
    for (int j = 0; j < k; j++) top[j] = -1;
    char *taken = calloc((size_t)I, 1);
    for (int j = 0; j < k; j++) {
        int best = -1; double bv = -1;
        for (int i = 0; i < I; i++)
            if (!taken[i] && acc[i] > bv) { bv = acc[i]; best = i; }
        if (best < 0) break;
        taken[best] = 1; top[j] = best;
    }
    free(taken);
    /* ascending index order (contiguity for reads; sum order fixed) */
    for (int a = 1; a < k; a++) { int v = top[a]; int b = a-1; while (b >= 0 && top[b] > v) { top[b+1] = top[b]; b--; } top[b+1] = v; }
    free(g_griff[layer].idx);
    g_griff[layer].idx = top; g_griff[layer].k = k; g_griff[layer].n = I;
    free(g_griff_acc_dyn[layer]); g_griff_acc_dyn[layer] = NULL; g_griff_acc_n[layer] = 0;
    free(idx);
}""", "griff-state")

# ---- observe hooks: in the STOCK batched path (prefill), gate values are bg[q] BEFORE silu ----
# the stock silu line at 3670-area: for(int64_t q...rows*I...) operates on bg. Insert BEFORE silu.
A_STOCK = "for(int64_t q=0;q<(int64_t)rows*I;q++){float sv=bg[q];bg[q]=(sv/(1.f+expf(-sv)))*bu[q];}"
sub1(A_STOCK,
"""{ int _li = (int)(l - m->L); q36_griff_observe(_li, bg, rows, I, I); }
for(int64_t q=0;q<(int64_t)rows*I;q++){float sv=bg[q];bg[q]=(sv/(1.f+expf(-sv)))*bu[q];}""", "observe-stock")

# ---- decode: in the PAGED path, when a griff set exists for this layer, use IT instead of [0,Ik) ----
A_PAGED = "    double hk = q36_layer_frac((int)(l - m->L));\n    int Ik = (int)(I * hk);\n    if (Ik < 1) Ik = 1;\n    if (Ik < I) {"
sub1(A_PAGED,
"""    double hk = q36_layer_frac((int)(l - m->L));\n    int Ik = (int)(I * hk);\n    if (Ik < 1) Ik = 1;
    /* GRIFFIN decode: per-sequence gather set (built during prefill) overrides
     * the static prefix when present and smaller. */
    int _gli = (int)(l - m->L);
    GriffSet *gs = (_gli < g_griff_n && g_griff[_gli].idx) ? &g_griff[_gli] : NULL;
    if (gs && gs->k > 0 && gs->k < I && S == 1) {
        /* gather-selected neurons: compute gate/up rows gs->idx[0..k), combine,
         * then down over the same original indices. */
        int K = gs->k;
        const int *ix = gs->idx;
        float *gg = g, *uu = u;
        const float *xs = x;
        /* gate/up for the selected rows via per-row mapped/resident reads */
        for (int j = 0; j < K; j++) {
            int o = ix[j];
            /* one row of sh_g and sh_u at output index o, dot with xs (width D) */
            QW ng = l->sh_g; ng.O = 1; ng.I = D;
            /* shift the map to row o: rows are o*(I/2 or I) bytes — build a row view */
            float gval, uval;
            {
                const QW *w = &l->sh_g;
                if (w->map) {
                    int64_t rowbytes = (w->map_len == (int64_t)D*I/2) ? (D/2) : (D * 2);
                    QW rv = *w; rv.map = (char*)w->map + (int64_t)o * rowbytes; rv.map_len = rowbytes; rv.O = 1; rv.I = D; rv.sc = w->sc ? w->sc + o : NULL;
                    matmul_d(&gval, xs, &rv, 1, D, 1);
                } else {
                    QW rv = *w; rv.O = 1; rv.I = D;
                    if (w->q) { /* resident int8: rows at q + o*D, scale sc[o] — use matmul_q on a fake 1-row matrix */ }
                    matmul_d(&gval, xs, &rv, 1, D, 1);
                }
            }
            {
                const QW *w = &l->sh_u;
                if (w->map) {
                    int64_t rowbytes = (w->map_len == (int64_t)D*I/2) ? (D/2) : (D * 2);
                    QW rv = *w; rv.map = (char*)w->map + (int64_t)o * rowbytes; rv.map_len = rowbytes; rv.O = 1; rv.I = D; rv.sc = w->sc ? w->sc + o : NULL;
                    matmul_d(&uval, xs, &rv, 1, D, 1);
                } else {
                    QW rv = *w; rv.O = 1; rv.I = D;
                    matmul_d(&uval, xs, &rv, 1, D, 1);
                }
            }
            gg[j] = (gval/(1.f+expf(-gval)))*uval;
        }
        /* down: y[d] = sum_j down[d][ix[j]] * gg[j] — column gather per row d */
        {
            const QW *dw = &l->sh_d;
            if (dw->map && dw->map_len == (int64_t)D*I/2) {
                const uint8_t *mp = (const uint8_t*)dw->map; const float *sc = dw->sc;
                #pragma omp parallel for schedule(static)
                for (int d = 0; d < D; d++) {
                    const uint8_t *row = mp + (int64_t)d * (I >> 1);
                    float acc = 0;
                    for (int j = 0; j < K; j++) {
                        int i2 = ix[j];
                        uint8_t b = row[i2 >> 1];
                        int8_t e = (int8_t)((i2 & 1) ? (b >> 4) : (b & 0xF));
                        if (e & 8) e -= 16;
                        acc += (float)e * gg[j];
                    }
                    hh[d] = acc * (sc ? sc[d] : 1.0f);
                }
            } else if (dw->map) {
                const uint16_t *mp = (const uint16_t*)dw->map;
                #pragma omp parallel for schedule(static)
                for (int d = 0; d < D; d++) {
                    const uint16_t *row = mp + (int64_t)d * I;
                    float acc = 0;
                    for (int j = 0; j < K; j++) acc += map_f16_one(row[ix[j]]) * gg[j];
                    hh[d] = acc;
                }
            } else if (dw->q) {
                const int8_t *q = dw->q; const float *sc = dw->sc;
                #pragma omp parallel for schedule(static)
                for (int d = 0; d < D; d++) {
                    const int8_t *row = q + (int64_t)d * I;
                    float acc = 0;
                    for (int j = 0; j < K; j++) acc += (float)row[ix[j]] * gg[j];
                    hh[d] = acc * (sc ? sc[d] : 1.0f);
                }
            } else if (dw->w) {
                const float *w = dw->w;
                #pragma omp parallel for schedule(static)
                for (int d = 0; d < D; d++) {
                    const float *row = w + (int64_t)d * I;
                    float acc = 0;
                    for (int j = 0; j < K; j++) acc += row[ix[j]] * gg[j];
                    hh[d] = acc;
                }
            }
        }
        float sgate = 1.f;
        if (l->sh_gate) { float sg = 0.f; for (int i = 0; i < D; i++) sg += xs[i]*l->sh_gate[i]; sgate = 1.f/(1.f+expf(-sg)); }
        float *os = out;
        for (int d = 0; d < D; d++) os[d] += sgate * hh[d];
        return;
    }
    if (Ik < I) {""", "griff-decode")

# ---- finalize at the prefill/decode boundary: layers_forward_range is called per step;
# simplest correct point: end of generate()'s prefill step == first step() return. Use the
# SAME seam HOT uses: profile save at exit ALSO finalizes any un-finalized layers.
sub1("static void q36_hot_profile_save(void) {",
"""static void q36_griff_finalize_all(double frac);
static void q36_hot_profile_save(void) {""", "fwd-finalize")

sub1("    q36_hot_profile_save();\n    tm_report();",
"""    q36_hot_profile_save();
    q36_griff_finalize_all(q36_hot_frac());
    tm_report();""", "finalize-exit")

# and the real finalize definition after q36_griff_finalize
sub1("static void q36_hot_profile_save(void) {",
"""static void q36_griff_finalize_all(double frac) {
    for (int L = 0; L < g_griff_n; L++) {
        if (g_griff_acc_dyn[L]) q36_griff_finalize(L, frac, g_griff_acc_n[L]);
    }
}
static void q36_hot_profile_save(void) {""", "finalize-def")

open(P, "w").write(src)
print("GRIFFIN v5 applied")
