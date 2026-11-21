/* mapi4_parity.c — gate for engine/mapi4_avx2.h (the vectorized matmul_map_i4 kernels).
 *
 *   gcc -O2 -fopenmp -Iengine -o /tmp/m4p engine/mapi4_parity.c -lm && /tmp/m4p
 *   /tmp/m4p bench          # timings at the 27B shapes (no pass/fail)
 *
 * 1. TIER 1 IS BIT-EXACT: q36_m4_stage_avx2 must produce the identical float
 *    array as the scalar loop (memcmp, not tolerance) for every I (incl. odd
 *    tails and I < 32), scale (1.0, normal, negative, tiny), unaligned output,
 *    and for EVERY one of the 256 byte values. The engine's dot_f32_lanes is
 *    untouched, so equal staging => equal outputs.
 * 2. TIER 2 IS ACCURATE: integer-dot vs a double-precision reference, relative
 *    RMS error under a bound, on Gaussian and outlier-heavy activations.
 * 3. TIER 2 IS DETERMINISTIC AND BATCH-INDEPENDENT: a token's result is
 *    bit-identical whether it is computed alone or inside a batch of 1..9, and
 *    across OpenMP thread counts.
 * Exit status is non-zero on any failure; "SKIP" (no AVX2) exits 0 loudly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mapi4_avx2.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL  " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static uint32_t rng_s = 0x9E3779B9u;
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5; return rng_s; }
static float gauss(void) {
    double u1 = (rnd() + 1.0) / 4294967297.0, u2 = (rnd() + 1.0) / 4294967297.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int have_avx2(void) {
#ifdef Q36_M4_X86
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

/* ---- 1. staging is bit-exact ---- */
static void test_stage(void) {
#ifdef Q36_M4_X86
    const int Is[] = { 2, 4, 6, 30, 32, 34, 62, 64, 66, 96, 130, 256, 1000, 5120, 17408 };
    const float rss[] = { 1.0f, 0.0123f, -3.5f, 1e-30f, 6.25e4f };
    long cases = 0;
    for (size_t a = 0; a < sizeof Is / sizeof *Is; a++) {
        int I = Is[a];
        uint8_t *row = malloc(I / 2 + 64);
        float *s0 = malloc(sizeof(float) * (I + 16)), *s1 = malloc(sizeof(float) * (I + 16));
        for (int k = 0; k < I / 2; k++) row[k] = (uint8_t)rnd();
        for (size_t b = 0; b < sizeof rss / sizeof *rss; b++)
            for (int off = 0; off < 3; off++) {            /* unaligned destination */
                memset(s0, 0xA5, sizeof(float) * (I + 16)); memset(s1, 0xA5, sizeof(float) * (I + 16));
                q36_m4_stage_scalar(s0 + off, row, I, rss[b]);
                q36_m4_stage_avx2  (s1 + off, row, I, rss[b]);
                /* the whole buffer incl. guard words: no overrun either */
                CHECK(memcmp(s0, s1, sizeof(float) * (I + 16)) == 0, "stage I=%d rs=%g off=%d differs", I, rss[b], off);
                cases++;
            }
        free(row); free(s0); free(s1);
    }
    /* every byte value (all 256 nibble pairs) */
    uint8_t all[256]; for (int k = 0; k < 256; k++) all[k] = (uint8_t)k;
    float e0[512], e1[512];
    q36_m4_stage_scalar(e0, all, 512, 1.0f); q36_m4_stage_avx2(e1, all, 512, 1.0f);
    CHECK(memcmp(e0, e1, sizeof e0) == 0, "stage: all 256 byte values differ");
    for (int k = 0; k < 16; k++) {   /* sanity-pin the encoding itself: low nibble = even element, two's complement */
        int lo = k < 8 ? k : k - 16;
        CHECK(e0[2 * k] == (float)lo, "encoding: byte %d low nibble -> %g want %d", k, e0[2 * k], lo);
    }
    printf("ok    tier 1 staging bit-identical to scalar (%ld shape/scale/offset cases + all 256 bytes)\n", cases);
#endif
}

/* ---- 1b. prefix staging (odd Ik) is bit-exact with the legacy down_proj loop ---- */
static void legacy_prefix(float *st, const uint8_t *row, int Ik, float rs) {
    for (int i = 0; i < Ik; i++) {
        uint8_t b = row[i >> 1];
        int8_t e = (int8_t)((i & 1) ? (b >> 4) : (b & 0xF));
        if (e & 8) e -= 16;
        st[i] = (float)e;
    }
    if (rs != 1.0f) for (int i = 0; i < Ik; i++) st[i] *= rs;
}
static void test_prefix(void) {
    const int Iks[] = {1, 2, 3, 5, 31, 32, 33, 63, 65, 1000, 1001, 2959, 2960, 4352, 17407};
    const float rss[] = {1.0f, 0.0123f, -3.5f, 6.25e4f};
    long cases = 0;
    for (size_t a = 0; a < sizeof Iks / sizeof *Iks; a++) {
        int Ik = Iks[a];
        uint8_t *row = malloc((Ik + 1) / 2 + 64);
        float *s0 = malloc(sizeof(float) * (Ik + 16)), *s1 = malloc(sizeof(float) * (Ik + 16));
        for (int k = 0; k < (Ik + 1) / 2 + 8; k++) row[k] = (uint8_t)rnd();
        for (size_t b = 0; b < sizeof rss / sizeof *rss; b++) {
            memset(s0, 0xA5, sizeof(float) * (Ik + 16)); memset(s1, 0xA5, sizeof(float) * (Ik + 16));
            legacy_prefix(s0, row, Ik, rss[b]); q36_m4_stage_prefix(s1, row, Ik, rss[b]);
            CHECK(memcmp(s0, s1, sizeof(float) * (Ik + 16)) == 0, "prefix Ik=%d rs=%g differs", Ik, rss[b]);
            cases++;
        }
        free(row); free(s0); free(s1);
    }
    printf("ok    prefix staging (odd/even Ik) bit-identical to the legacy down_proj loop (%ld cases)\n", cases);
}

/* ---- 2/3. integer dot ---- */
static void ref_matmul(double *y, const float *x, const uint8_t *W, const float *sc, int S, int I, int O) {
    float *st = malloc(sizeof(float) * I);
    for (int o = 0; o < O; o++) {
        q36_m4_stage_scalar(st, W + (int64_t)o * (I / 2), I, 1.0f);   /* exact integers */
        for (int s = 0; s < S; s++) {
            double a = 0; for (int i = 0; i < I; i++) a += (double)st[i] * x[(int64_t)s * I + i];
            y[(int64_t)s * O + o] = a * (sc ? sc[o] : 1.0f);
        }
    }
    free(st);
}

static double relrms(const float *y, const double *r, int n) {
    double e = 0, d = 0; for (int i = 0; i < n; i++) { double t = y[i] - r[i]; e += t * t; d += r[i] * r[i]; }
    return sqrt(e / (d > 0 ? d : 1));
}

static void test_idot(void) {
#ifdef Q36_M4_X86
    const int I = 1024, O = 96, SMAX = 9;
    uint8_t *W = malloc((size_t)O * I / 2); for (size_t k = 0; k < (size_t)O * I / 2; k++) W[k] = (uint8_t)rnd();
    float *sc = malloc(sizeof(float) * O); for (int o = 0; o < O; o++) sc[o] = 0.005f + 0.0001f * o;
    float *x = malloc(sizeof(float) * SMAX * I);
    float *y = malloc(sizeof(float) * SMAX * O), *yb = malloc(sizeof(float) * SMAX * O);
    double *r = malloc(sizeof(double) * SMAX * O);

    for (int kind = 0; kind < 2; kind++) {
        for (int i = 0; i < SMAX * I; i++) {
            float v = gauss();
            if (kind == 1 && (rnd() % 97) == 0) v *= 40.0f;     /* outlier-heavy, like real LLM activations */
            x[i] = v;
        }
        ref_matmul(r, x, W, sc, SMAX, I, O);
        q36_m4_idot_avx2(y, x, W, sc, SMAX, I, O);
        double e = relrms(y, r, SMAX * O);
        double bound = kind == 0 ? 0.02 : 0.05;
        printf("      idot relative RMS error vs f64 reference (%s activations): %.3f%%  (bound %.0f%%)\n",
               kind == 0 ? "gaussian" : "outlier-heavy", 100 * e, 100 * bound);
        CHECK(e < bound, "idot error %.4f exceeds bound %.2f (%s)", e, bound, kind == 0 ? "gaussian" : "outliers");
    }
    /* batch independence: token s alone == token s inside any batch */
    q36_m4_idot_avx2(y, x, W, sc, SMAX, I, O);
    int diff = 0;
    for (int S = 1; S <= SMAX; S++) {
        q36_m4_idot_avx2(yb, x, W, sc, S, I, O);
        for (int s = 0; s < S; s++)
            for (int o = 0; o < O; o++)
                if (memcmp(&yb[(int64_t)s * O + o], &y[(int64_t)s * O + o], 4)) diff++;
    }
    CHECK(diff == 0, "idot result depends on batch size (%d elements differ)", diff);
    if (!diff) printf("ok    tier 2 integer dot: token result independent of batch size (S=1..%d)\n", SMAX);
#ifdef _OPENMP
    int nt = omp_get_max_threads(); int d2 = 0;
    for (int t = 1; t <= 4; t++) {
        omp_set_num_threads(t); q36_m4_idot_avx2(yb, x, W, sc, SMAX, I, O);
        d2 += memcmp(yb, y, sizeof(float) * SMAX * O) != 0;
    }
    omp_set_num_threads(nt);
    CHECK(d2 == 0, "idot result depends on thread count");
    if (!d2) printf("ok    tier 2 integer dot: bit-identical across 1..4 threads\n");
#endif
    free(W); free(sc); free(x); free(y); free(yb); free(r);
#endif
}

/* ---- benchmark ---- */
static void bench_shape(const char *name, int I, int O, int reps) {
    uint8_t *W = malloc((size_t)O * I / 2); for (size_t k = 0; k < (size_t)O * I / 2; k++) W[k] = (uint8_t)rnd();
    float *sc = malloc(sizeof(float) * O); for (int o = 0; o < O; o++) sc[o] = 0.01f;
    float *x = malloc(sizeof(float) * I), *st = malloc(sizeof(float) * I), *y = malloc(sizeof(float) * O);
    for (int i = 0; i < I; i++) x[i] = gauss();
    volatile float sink = 0;
    double w = (double)reps * O * I;
    double t0 = now();
    for (int r = 0; r < reps; r++) for (int o = 0; o < O; o++) { q36_m4_stage_scalar(st, W + (int64_t)o * (I/2), I, sc[o]); sink += st[r % I]; }
    double ts = now() - t0; t0 = now();
    for (int r = 0; r < reps; r++) for (int o = 0; o < O; o++) { q36_m4_stage_avx2(st, W + (int64_t)o * (I/2), I, sc[o]); sink += st[r % I]; }
    double tv = now() - t0; t0 = now();
    for (int r = 0; r < reps; r++) q36_m4_idot_avx2(y, x, W, sc, 1, I, O);
    double ti = now() - t0;
    printf("%-34s staging only: scalar %.3f -> avx2 %.3f ns/weight (%.1fx) | idot %.3f ns/weight total (stage+dot of the old path is higher)\n",
           name, ts / w * 1e9, tv / w * 1e9, ts / tv, ti / w * 1e9);
    free(W); free(sc); free(x); free(st); free(y); (void)sink;
}

int main(int argc, char **argv) {
    if (!have_avx2()) { printf("SKIP  this CPU has no AVX2+FMA; the engine falls back to the scalar loop\n"); return 0; }
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        bench_shape("L2-resident (I=5120, O=128)", 5120, 128, 2000);
        bench_shape("gate/up shape (I=5120, O=17408)", 5120, 17408, 6);
        bench_shape("down shape (I=17408, O=5120)", 17408, 5120, 6);
        return 0;
    }
    test_stage();
    test_prefix();
    test_idot();
    if (fails) { printf("mapi4 gate: FAIL (%d)\n", fails); return 1; }
    printf("mapi4 gate: PASS\n");
    return 0;
}
