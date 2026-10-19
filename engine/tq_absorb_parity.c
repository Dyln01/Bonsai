/* tq_absorb_parity.c — Phase 3b-rot numeric gate (pure C99, no container).
 *
 * Proves the absorbed-rotation math equals the reference math for codec 1:
 *   score:  q·dequant(row)  ==  rotate(q)·levels(row)          (radius hoisted)
 *   accum:  Σ a_t dequant(V_t)  ==  unrotate(Σ a_t levels(V_t))
 * up to fp re-association (tol 1e-3 rel on the dot, elements 1e-3).
 * Build: gcc -O2 -Iengine -o /tmp/absorb engine/tq_absorb_parity.c -lm && /tmp/absorb
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kv_tq.h"

static unsigned long st = 88172645463325252ul;
static double rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (double)(st >> 11) / 9007199254740992.0; }
static float gauss(void) { double u = rnd() + 1e-12, v = rnd(); return (float)(sqrt(-2*log(u)) * cos(6.283185307179586*v)); }

static int fail = 0;
static void check(double got, double want, double tol, const char *what) {
    double d = fabs(got - want), s = fabs(want) > 1e-30 ? fabs(want) : 1.0;
    if (d / s > tol) { printf("FAIL %s: got %.9g want %.9g (rel %.3g)\n", what, got, want, d/s); fail = 1; }
}

int main(void) {
    const int NS[2] = { 64, 128 };
    for (int ni = 0; ni < 2; ni++) {
        int n = NS[ni];
        float *row = malloc(n * sizeof(float)), *rec = malloc(n * sizeof(float));
        float *q = malloc(n * sizeof(float)), *qr = malloc(n * sizeof(float));
        float *lev = malloc(n * sizeof(float)), *acc = malloc(n * sizeof(float));
        uint8_t *pk = malloc((size_t)(n / 2) + 8);
        for (int i = 0; i < n; i++) { row[i] = gauss(); q[i] = gauss(); }
        float radius = coli_q4_quant_row(row, pk, n);
        /* reference: full dequant (levels + unrotate), radius included */
        coli_q4_dequant_row(pk, radius, rec, n);
        double sref = 0;
        for (int i = 0; i < n; i++) sref += (double)q[i] * rec[i];
        /* absorbed: rotate the query, dot against rotated-space levels (radius hoisted) */
        coli_tq_rotate(q, qr, n, COLI_TQ_SEED);
        {
            float std = radius / sqrtf((float)n);
            for (int i = 0; i < n; i++) {
                int c = (pk[i >> 1] >> ((i & 1) * 4)) & 0xF;
                lev[i] = coli_q4_lev[c] * std;
            }
        }
        double sabs = 0;
        for (int i = 0; i < n; i++) sabs += (double)qr[i] * lev[i];
        check(sabs, sref, 1e-3, "score n=%d");
        /* V accumulation: 12 rows, random weights */
        float *accr = calloc(n, sizeof(float));
        memset(acc, 0, n * sizeof(float));
        for (int t = 0; t < 12; t++) {
            float a = (float)(rnd() - 0.5);
            float vr[256], vrec[256]; uint8_t vp[136];
            for (int i = 0; i < n; i++) vr[i] = gauss();
            float vrad = coli_q4_quant_row(vr, vp, n);
            coli_q4_dequant_row(vp, vrad, vrec, n);
            for (int i = 0; i < n; i++) accr[i] += a * vrec[i];
            float vstd = vrad / sqrtf((float)n);
            for (int i = 0; i < n; i++) {
                int c = (vp[i >> 1] >> ((i & 1) * 4)) & 0xF;
                acc[i] += a * coli_q4_lev[c] * vstd;
            }
        }
        coli_tq_unrotate(acc, n, COLI_TQ_SEED);
        double nr = 0, nd = 0;
        for (int i = 0; i < n; i++) { nr += (double)accr[i]*accr[i]; nd += (double)(acc[i]-accr[i])*(acc[i]-accr[i]); }
        if (sqrt(nd) / (sqrt(nr) + 1e-30) > 1e-3) { printf("FAIL accum rel L2 %.4g (n=%d)\n", sqrt(nd)/sqrt(nr), n); fail = 1; }
        free(row); free(rec); free(q); free(qr); free(lev); free(acc); free(pk); free(accr);
    }
    if (fail) { printf("ABSORB PARITY FAIL\n"); return 1; }
    printf("ABSORB PARITY OK: score + rotated-space accumulation match reference (n=64,128)\n");
    return 0;
}
