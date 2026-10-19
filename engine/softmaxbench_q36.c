/* softmaxbench_q36.c — is a fast exponential worth shipping in softmax_row?
 *
 * Stock softmax_row is 3 passes over the O(T) score row with a libm expf call
 * per element (nothing vectorizes it without -ffast-math). Candidates:
 *   sfast : same 3 passes, scalar q36_fexp (exponent bit-trick + degree-5 poly),
 *           final pass x*(1/s) instead of x/s.
 *   vfast : AVX2 vector q36_fexp8 + vector max pass (the engine builds with
 *           -march=x86-64-v3, so AVX2+FMA are compiled in).
 * Gates printed before any timing:
 *   max rel err of q36_fexp vs expf on a dense grid over [-104, 0]
 *   (the post-max-subtraction domain) and the softmax-domain consequence:
 *   worst elementwise |p_fast - p_stock| and the sum, per distribution.
 * Also times a silu-shaped arm (expf(-x) per element over 4096-wide rows) to
 * size the follow-up opportunity in the MLP path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + 1e-9*ts.tv_nsec; }

/* ---- the candidate ---- */
static inline float q36_fexp(float x) {
    if (x < -87.0f) return 0.f;                  /* exp < 1.6e-38: below f32 normal range.
                                                    * ALSO: the bit trick needs n >= -126
                                                    * (n+127 >= 1) or the exponent field
                                                    * wraps — flush, never wrap. */
    const float L_HI = 1.4426950216293335f;      /* f32-nearest log2(e) */
    const float L_LO = 1.925963e-8f;            /* log2(e) - L_HI */
    float t = x * L_HI;                          /* coarse: only to pick n */
    float n = (float)(int)(t + (t < 0.f ? -0.5f : 0.5f));
    float f = fmaf(x, L_LO, fmaf(x, L_HI, -n));  /* exact fraction: 2 FMAs, no |t|-amplified rounding */
    float p = 1.0000001f + f * (0.6931472f + f * (0.24022107f +
              f * (0.05550327f + f * (0.009676037f + f * 0.0013400433f))));
    union { uint32_t i; float f; } u;
    u.i = (uint32_t)((int)n + 127) << 23;
    return p * u.f;
}

#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 q36_fexp8(__m256 x) {
    const __m256 FLOOR = _mm256_set1_ps(-87.0f);   /* n+127 >= 1: the exponent field
                                                      * must never wrap */
    const __m256 L_HI = _mm256_set1_ps(1.4426950216293335f), L_LO = _mm256_set1_ps(1.925963e-8f);
    const __m256 C0 = _mm256_set1_ps(1.0000001f), C1 = _mm256_set1_ps(0.6931472f);
    const __m256 C2 = _mm256_set1_ps(0.24022107f), C3 = _mm256_set1_ps(0.05550327f);
    const __m256 C4 = _mm256_set1_ps(0.009676037f), C5 = _mm256_set1_ps(0.0013400433f);
    const __m256i BIAS = _mm256_set1_epi32(127);
    __m256 mask = _mm256_cmp_ps(x, FLOOR, _CMP_LT_OQ);
    x = _mm256_max_ps(x, FLOOR);                        /* clamp once */
    __m256 t = _mm256_mul_ps(x, L_HI);                  /* coarse: to pick n */
    __m256 rnd = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_fmadd_ps(x, L_LO, _mm256_fmsub_ps(x, L_HI, rnd));   /* exact fraction */
    __m256 p = C5;
    p = _mm256_fmadd_ps(f, p, C4);
    p = _mm256_fmadd_ps(f, p, C3);
    p = _mm256_fmadd_ps(f, p, C2);
    p = _mm256_fmadd_ps(f, p, C1);
    p = _mm256_fmadd_ps(f, p, C0);
    __m256i e = _mm256_add_epi32(_mm256_cvtps_epi32(rnd), BIAS);
    __m256 two_n = _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
    __m256 r = _mm256_mul_ps(p, two_n);
    return _mm256_andnot_ps(mask, r);                   /* flushed lanes -> 0 */
}
#endif

/* ---- the implementations under test ---- */
static void sm_stock(float *x, int n) {                 /* verbatim engine copy */
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}
static void sm_sfast(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { float e = q36_fexp(x[i]-m); x[i] = e; s += e; }
    float inv = 1.f / s; for (int i = 0; i < n; i++) x[i] *= inv;
}
#if defined(__AVX2__) && defined(__FMA__)
static void sm_vfast(float *x, int n) {
    const __m256 VM = _mm256_set1_ps(-1e30f);
    __m256 vm = VM;
    int i = 0;
    for (; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(x + i));
    float m = -1e30f;
    /* horizontal max of vm */
    __m128 lo = _mm256_castps256_ps128(vm), hi = _mm256_extractf128_ps(vm, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 1));
    m = _mm_cvtss_f32(lo);
    for (; i < n; i++) if (x[i] > m) m = x[i];
    __m256 vs = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 e = q36_fexp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(m)));
        _mm256_storeu_ps(x + i, e);
        vs = _mm256_add_ps(vs, e);
    }
    float s = 0;
    __m128 slo = _mm256_castps256_ps128(vs), shi = _mm256_extractf128_ps(vs, 1);
    __m128 sd = _mm_add_ps(slo, shi);
    sd = _mm_add_ps(sd, _mm_movehl_ps(sd, sd));
    sd = _mm_add_ss(sd, _mm_shuffle_ps(sd, sd, 1));
    s = _mm_cvtss_f32(sd);
    for (; i < n; i++) { float e = q36_fexp(x[i]-m); x[i] = e; s += e; }
    float inv = 1.f / s;
    for (i = 0; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(inv)));
    for (; i < n; i++) x[i] *= inv;
}
#endif

/* ---- gates ---- */
static uint32_t rng = 0x9E3779B9u;
static float frand(void){ rng = rng*1664525u+1013904223u; return (float)((rng>>8)&0xFFFFFF)/(float)0x1000000 * 2.f - 1.f; }
static float gauss(void){ float u=(frand()+1.f)*.5f+1e-7f, v=(frand()+1.f)*.5f; return sqrtf(-2.f*logf(u))*cosf(2.f*3.14159265f*v); }

/* E6 retry: ONLINE (flash-shaped) softmax — per-element expf vs q36_fexp with
 * the scalar block-rescale kept exact. Models the flash loop's exp count:
 * one expf per (row, t) plus one exact rescale exp per block. */
static void on_stock(const float *sc, int n, int TB, float *lout) {
    float m = -1e30f, l = 0.f;
    for (int t0 = 0; t0 < n; t0 += TB) {
        int tn = n - t0 < TB ? n - t0 : TB;
        float m2 = -1e30f;
        for (int t = 0; t < tn; t++) if (sc[t0+t] > m2) m2 = sc[t0+t];
        l *= expf(m - m2); m = m2;
        for (int t = 0; t < tn; t++) l += expf(sc[t0+t] - m2);
    }
    *lout = l;
}
static void on_fast(const float *sc, int n, int TB, float *lout) {
    float m = -1e30f, l = 0.f;
    for (int t0 = 0; t0 < n; t0 += TB) {
        int tn = n - t0 < TB ? n - t0 : TB;
        float m2 = -1e30f;
        for (int t = 0; t < tn; t++) if (sc[t0+t] > m2) m2 = sc[t0+t];
        l *= expf(m - m2); m = m2;                    /* scalar rescale stays exact */
        for (int t = 0; t < tn; t++) l += q36_fexp(sc[t0+t] - m2);
    }
    *lout = l;
}

int main(void) {
    /* gate 1: exp accuracy over the softmax domain [-104, 0] */
    double worst = 0; float wx = 0;
    for (double d = -104.0; d <= 0.0; d += 0.001) {
        float x = (float)d, e = expf(x), f = q36_fexp(x);
        if (e > 1e-30f) { double r = fabs((double)f/e - 1.0); if (r > worst) { worst = r; wx = x; } }
    }
    printf("[gate] q36_fexp max rel err vs expf over [-104,0]: %.3e (at x=%.3f)\n", worst, wx);
    if (worst > 3e-6) { printf("GATE FAILED: rel err too large\n"); return 1; }

    /* gate 1b: the underflow zone must flush to exact zero, never wrap the
     * exponent field (the N(0,15) arm caught exactly this class of bug) */
    for (double d = -104.0; d < -87.0; d += 0.0013) {
        float f = q36_fexp((float)d);
        if (!(f >= 0.f && f <= 1.6e-38f)) { printf("GATE FAILED: underflow zone x=%.3f -> %.3e\n", d, f); return 1; }
    }
    printf("[gate] underflow zone [-104,-87): all outputs in [0, 1.6e-38]\n");

    /* gate 2: softmax-domain consequence vs an f64 REFERENCE (not vs stock —
     * stock's plain f32 accumulator drifts ~1e-5 at 16k; the fast path's
     * 2-lane accumulation is more accurate, and the gate must not punish it) */
    const int T = 16384;
    float *a = malloc((size_t)T*sizeof(float)), *b = malloc((size_t)T*sizeof(float));
    float sigmas[2] = {3.f, 15.f};
    for (int sgi = 0; sgi < 2; sgi++) {
        double *ref = malloc((size_t)T*sizeof(double));
        double m2 = -1e30, s2 = 0;
        for (int i = 0; i < T; i++) { a[i] = b[i] = gauss()*sigmas[sgi];
            if (a[i] > m2) m2 = a[i]; }
        for (int i = 0; i < T; i++) { ref[i] = exp((double)a[i]-m2); s2 += ref[i]; }
        for (int i = 0; i < T; i++) ref[i] /= s2;
        sm_stock(a, T);
#if defined(__AVX2__) && defined(__FMA__)
        sm_vfast(b, T);
#else
        sm_sfast(b, T);
#endif
        double e_stock = 0, e_fast = 0, sa = 0, sb = 0;
        for (int i = 0; i < T; i++) {
            double ds = fabs((double)a[i]-ref[i]), df = fabs((double)b[i]-ref[i]);
            if (ds > e_stock) e_stock = ds;
            if (df > e_fast) e_fast = df;
            sa += a[i]; sb += b[i];
        }
        printf("[gate] N(0,%.0f): max|p-ref|  stock %.3e  fast %.3e | sums stock %.9f fast %.9f\n",
               sigmas[sgi], e_stock, e_fast, sa, sb);
        if (e_fast > 3e-6 || e_fast > e_stock * 1.5) { printf("GATE FAILED\n"); return 1; }
        free(ref);
    }

    /* timing */
    printf("\n%-8s %10s %14s %14s %14s\n", "T", "stock ns/row", "sfast ns/row", "vfast ns/row", "speedup");
    int Ts[] = {1024, 4096, 8192, 16384, 32768};
    for (unsigned ti = 0; ti < sizeof Ts/sizeof Ts[0]; ti++) {
        int n = Ts[ti];
        int reps = 40000000 / n;
        double t0, t1, t2;
        /* warm */
        for (int i = 0; i < n; i++) a[i] = gauss()*5.f;
        t0 = now_s();
        for (int r = 0; r < reps; r++) { a[0] += 1e-3f; sm_stock(a, n); }
        t1 = now_s();
        for (int r = 0; r < reps; r++) { a[0] += 1e-3f; sm_sfast(a, n); }
        t2 = now_s();
        double s_stock = (t1-t0)/reps*1e9, s_sfast = (t2-t1)/reps*1e9;
#if defined(__AVX2__) && defined(__FMA__)
        double t3;
        t2 = now_s();
        for (int r = 0; r < reps; r++) { a[0] += 1e-3f; sm_vfast(a, n); }
        t3 = now_s();
        double s_vfast = (t3-t2)/reps*1e9;
        printf("%-8d %12.0f %14.0f %14.0f %13.2fx\n", n, s_stock, s_sfast, s_vfast, s_stock/s_vfast);
#else
        printf("%-8d %12.0f %14.0f %14s %13.2fx\n", n, s_stock, s_sfast, "(no avx2)", s_stock/s_sfast);
#endif
    }

    /* silu-shaped follow-up sizing: expf(-x) vs fast, 4096-wide rows.
     * Each arm reads the SAME pristine input (a copy) — the first version
     * fed loop 2 the transformed output of loop 1 and measured noise. */
    {
        int n = 4096, reps = 4000;
        float *src = malloc((size_t)n*sizeof(float)), *out = malloc((size_t)n*sizeof(float));
        for (int i = 0; i < n; i++) src[i] = gauss()*4.f;
        double t0 = now_s();
        for (int r = 0; r < reps; r++)
            for (int i = 0; i < n; i++) { float sv = src[i]; out[i] = (sv/(1.f+expf(-sv)))*0.5f; }
        double t1 = now_s();
        for (int r = 0; r < reps; r++)
            for (int i = 0; i < n; i++) { float sv = src[i]; out[i] = (sv/(1.f+q36_fexp(-sv)))*0.5f; }
        double t2 = now_s();
        /* worst activation delta on this input */
        double wact = 0;
        for (int i = 0; i < n; i++) {
            float a1 = (src[i]/(1.f+expf(-src[i])))*0.5f, a2 = (src[i]/(1.f+q36_fexp(-src[i])))*0.5f;
            double d = fabs((double)a1-a2); if (d > wact) wact = d;
        }
        printf("\n[silu-shaped] expf: %.1f ns/row(4096)  fast: %.1f  -> %.2fx in the MLP gate | worst |act delta| %.2e\n",
               (t1-t0)/reps*1e9, (t2-t1)/reps*1e9, (t1-t0)/(t2-t1), wact);
        free(src); free(out);
    }
    /* E6 retry measurement + error gate (f64 reference) */
    {
        int T2 = 16384, TB2 = 256, reps = 2000;
        float *sc2 = malloc((size_t)T2 * sizeof(float)); float l1, l2;
        for (int i = 0; i < T2; i++) sc2[i] = -fabsf(gauss()) * 4.f;
        double mx = -1e30; for (int i = 0; i < T2; i++) if (sc2[i] > mx) mx = sc2[i];
        double ref = 0; for (int i = 0; i < T2; i++) ref += exp((double)sc2[i] - mx);
        on_stock(sc2, T2, TB2, &l1); on_fast(sc2, T2, TB2, &l2);
        double e1 = fabs((double)l1/ref - 1.0), e2 = fabs((double)l2/ref - 1.0);
        double t0 = now_s(); for (int r = 0; r < reps; r++) on_stock(sc2, T2, TB2, &l1); double t1 = now_s();
        for (int r = 0; r < reps; r++) on_fast(sc2, T2, TB2, &l2); double t2 = now_s();
        printf("[online-softmax E6] stock %.0f ns/row(16k) | fast-exp+exact-rescale %.0f -> %.2fx | rel err vs f64: stock %.2e fast %.2e\n",
               (t1-t0)/reps*1e9, (t2-t1)/reps*1e9, (t1-t0)/(t2-t1), e1, e2);
        if (e2 > 4*e1 + 1e-6) { printf("GATE FAILED: online fast-exp error too large\n"); return 1; }
        free(sc2);
    }
    printf("BENCH DONE\n");
    return 0;
}
