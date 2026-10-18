#!/usr/bin/env python3
"""patch_pagebatch.py — batch the HOT_FRAC paged path across S rows (v3.2).

The paged branch looped per token (s) with S=1 matmul calls, re-dequantizing the
same narrowed rows S times during prefill — the exact waste MAPFAST removed from
the full path. Fix: narrow gate/up ONCE and call the batched matmul over all S
rows (chunks of B, mirroring qwen_shared_batch_rows), and a batched
q36_down_prefix. Identical math; Ik*I staging done once per row chunk.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
if "PAGEBATCH" in src:
    print("already patched"); sys.exit(0)

def sub1(anchor, repl, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag}: count={src.count(anchor)}"
    src = src.replace(anchor, repl)

# 1) batched down-prefix: takes [S, Ik] g, writes [S, D]
A_DP = "static void q36_down_prefix(const QW *dw, float *dst, const float *g, int D, int I, int Ik) {"
sub1(A_DP,
"""static void q36_down_prefix_batch(const QW *dw, float *dst, const float *g,
                                  int S, int D, int I, int Ik) {
    /* dst[s][d] = sum_{i<Ik} down[d][i] * g[s][i]; staged row dequant once per d. */
    if (dw->q) {
        const int8_t *q = dw->q; const float *sc = dw->sc;
        #pragma omp parallel
        {
            float *st = malloc((size_t)Ik * sizeof(float));
            if (!st) { fprintf(stderr, "OOM down stage\\n"); exit(1); }
            #pragma omp for schedule(static)
            for (int d = 0; d < D; d++) {
                const int8_t *row = q + (int64_t)d * I;
                for (int i = 0; i < Ik; i++) st[i] = (float)row[i];
                float rs = sc ? sc[d] : 1.0f;
                if (rs != 1.0f) for (int i = 0; i < Ik; i++) st[i] *= rs;
                for (int s = 0; s < S; s++)
                    dst[(int64_t)s*D + d] = dot_f32_lanes(g + (int64_t)s*Ik, st, Ik);
            }
            free(st);
        }
    } else if (dw->map && dw->map_len == (int64_t)D*I/2) {
        const uint8_t *mp = (const uint8_t*)dw->map; const float *sc = dw->sc;
        #pragma omp parallel
        {
            float *st = malloc((size_t)Ik * sizeof(float));
            if (!st) { fprintf(stderr, "OOM down stage\\n"); exit(1); }
            #pragma omp for schedule(static)
            for (int d = 0; d < D; d++) {
                const uint8_t *row = mp + (int64_t)d * (I >> 1);
                for (int i = 0; i < Ik; i++) {
                    uint8_t b = row[i >> 1];
                    int8_t e = (int8_t)((i & 1) ? (b >> 4) : (b & 0xF));
                    if (e & 8) e -= 16;
                    st[i] = (float)e;
                }
                float rs = sc ? sc[d] : 1.0f;
                if (rs != 1.0f) for (int i = 0; i < Ik; i++) st[i] *= rs;
                for (int s = 0; s < S; s++)
                    dst[(int64_t)s*D + d] = dot_f32_lanes(g + (int64_t)s*Ik, st, Ik);
            }
            free(st);
        }
    } else if (dw->map) {   /* f16 mapped */
        const uint16_t *mp = (const uint16_t*)dw->map;
        #pragma omp parallel
        {
            float *st = malloc((size_t)Ik * sizeof(float));
            if (!st) { fprintf(stderr, "OOM down stage\\n"); exit(1); }
            #pragma omp for schedule(static)
            for (int d = 0; d < D; d++) {
                const uint16_t *row = mp + (int64_t)d * I;
                for (int i = 0; i < Ik; i++) st[i] = map_f16_one(row[i]);
                for (int s = 0; s < S; s++)
                    dst[(int64_t)s*D + d] = dot_f32_lanes(g + (int64_t)s*Ik, st, Ik);
            }
            free(st);
        }
    } else if (dw->w) {
        const float *w = dw->w;
        #pragma omp parallel
        {
            float *st = malloc((size_t)Ik * sizeof(float));
            if (!st) { fprintf(stderr, "OOM down stage\\n"); exit(1); }
            #pragma omp for schedule(static)
            for (int d = 0; d < D; d++) {
                const float *row = w + (int64_t)d * I;
                for (int i = 0; i < Ik; i++) st[i] = row[i];
                for (int s = 0; s < S; s++)
                    dst[(int64_t)s*D + d] = dot_f32_lanes(g + (int64_t)s*Ik, st, Ik);
            }
            free(st);
        }
    }
}

static void q36_down_prefix(const QW *dw, float *dst, const float *g, int D, int I, int Ik) {""", "down-batch")

# 2) replace the per-token paged loop with a batched one
A_LOOP = """        for (int s = 0; s < S; s++) {
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
        return;"""
sub1(A_LOOP,
"""        {   /* PAGEBATCH: one narrowed batched matmul per chunk of B rows, staged
             * rows dequantized once and dotted for every row in the chunk. */
            int B = qwen_shared_batch_rows(S, D, Ik);
            float *bg = falloc((int64_t)B * Ik), *bu = falloc((int64_t)B * Ik);
            float *bh = falloc((int64_t)B * D);
            float *bp = falloc((int64_t)B * D);   /* paged down output */
            for (int base = 0; base < S; base += B) {
                int rows = S - base < B ? S - base : B;
                matmul_d(bg, x + (int64_t)base*D, &ng, rows, D, Ik);
                matmul_d(bu, x + (int64_t)base*D, &nu, rows, D, Ik);
                for (int64_t q = 0; q < (int64_t)rows*Ik; q++) {
                    float sv = bg[q]; bg[q] = (sv/(1.f+expf(-sv)))*bu[q];
                }
                q36_down_prefix_batch(&l->sh_d, bp, bg, rows, D, I, Ik);
                for (int s = 0; s < rows; s++) {
                    const float *xs = x + (int64_t)(base+s)*D;
                    float sgate = 1.f;
                    if (l->sh_gate) { float sg = 0.f; for (int i = 0; i < D; i++) sg += xs[i]*l->sh_gate[i]; sgate = 1.f/(1.f+expf(-sg)); }
                    float *os = out + (int64_t)(base+s)*D;
                    const float *ph = bp + (int64_t)s*D;
                    for (int d = 0; d < D; d++) os[d] += sgate * ph[d];
                }
            }
            free(bg); free(bu); free(bh); free(bp);
        }
        return;""", "page-batch")

open(P, "w").write(src)
print("PAGEBATCH v3.2 applied")
