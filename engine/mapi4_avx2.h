/* mapi4_avx2.h — vectorized kernels for matmul_map_i4 (the TRUNK_MAP int4 matmul).
 *
 * WHY. matmul_map_i4 stages each output row through a per-nibble scalar loop
 * (branchy sign-extend, int->float, then a separate scale pass) before the
 * fixed-lane dot. Measured in isolation that staging costs ~1 ns/weight
 * (~2 cycles/weight) — about 3.6 s per forward at the 27B operating point on
 * 4 cores even with every weight already in RAM. It is the next wall once
 * the page-cache thrash is fixed.
 *
 * TIER 1 (default, BIT-EXACT): q36_m4_stage_row() produces the SAME float
 * array the scalar loop produced — small exact integers, then one IEEE f32
 * multiply by the row scale — so feeding it to the unchanged dot_f32_lanes
 * gives bit-identical outputs. Only the cost of staging changes.
 *
 * TIER 2 (OPT-IN, Q36_MAPI4_IDOT=1, NOT bit-exact): q36_m4_idot() quantizes
 * the activations to int8 (one scale per 64 elements — the same grouping the
 * resident int4 kernel uses) and does the whole dot in integer SIMD
 * (vpmaddubsw), never materializing floats. ~an order of magnitude faster
 * again, but the activation rounding changes outputs slightly: validate
 * perplexity before trusting it. Needs I % 64 == 0, otherwise Tier 1.
 *
 * DISPATCH is at RUN time (__builtin_cpu_supports + function-level target
 * attributes), so it works whatever -march the engine was built with and
 * falls back to the scalar loop on a CPU without AVX2/FMA.
 *
 * KNOBS: Q36_MAPI4_SCALAR=1 forces the original scalar loop (A/B, escape
 *        hatch). Q36_MAPI4_IDOT=1 enables Tier 2.
 *
 * Packed layout (unchanged): row o = I/2 bytes; byte k holds element 2k in
 * the LOW nibble and element 2k+1 in the HIGH nibble, each a two's-complement
 * int4; element value = nibble * sc[o]. Verified by engine/mapi4_parity.c.
 */
#ifndef COLI_QWEN_MAPI4_AVX2_H
#define COLI_QWEN_MAPI4_AVX2_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define Q36_M4_X86 1
#endif

/* The original loop, byte for byte: the reference the vector path must equal. */
static inline void q36_m4_stage_scalar(float *st, const uint8_t *row, int I, float rs) {
    for (int i = 0; i < I; i += 2) {
        uint8_t b = row[i >> 1];
        int8_t e0 = (int8_t)(b & 0xF);  if (e0 & 8)  e0 -= 16;
        int8_t e1 = (int8_t)(b >> 4);   if (e1 & 8)  e1 -= 16;
        st[i] = (float)e0; st[i+1] = (float)e1;
    }
    if (rs != 1.0f) for (int i = 0; i < I; i++) st[i] *= rs;
}

#ifdef Q36_M4_X86
/* 32 elements (16 bytes) per iteration; any remainder goes through the same
 * scalar arithmetic, so every output element is computed identically. */
__attribute__((target("avx2,fma")))
static void q36_m4_stage_avx2(float *st, const uint8_t *row, int I, float rs) {
    const __m128i m4 = _mm_set1_epi8(0x0F), eight = _mm_set1_epi8(8);
    const __m256 vrs = _mm256_set1_ps(rs);
    const int scale = (rs != 1.0f);
    int i = 0;
    for (; i + 32 <= I; i += 32) {
        __m128i b  = _mm_loadu_si128((const __m128i *)(row + (i >> 1)));
        __m128i lo = _mm_and_si128(b, m4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m4);
        lo = _mm_sub_epi8(_mm_xor_si128(lo, eight), eight);   /* nibble -> int8: (n ^ 8) - 8 */
        hi = _mm_sub_epi8(_mm_xor_si128(hi, eight), eight);
        __m128i ea = _mm_unpacklo_epi8(lo, hi);               /* elements i    .. i+15 in order */
        __m128i eb = _mm_unpackhi_epi8(lo, hi);               /* elements i+16 .. i+31 in order */
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ea));
        __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(ea, 8)));
        __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(eb));
        __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(eb, 8)));
        if (scale) {
            f0 = _mm256_mul_ps(f0, vrs); f1 = _mm256_mul_ps(f1, vrs);
            f2 = _mm256_mul_ps(f2, vrs); f3 = _mm256_mul_ps(f3, vrs);
        }
        _mm256_storeu_ps(st + i,      f0); _mm256_storeu_ps(st + i + 8,  f1);
        _mm256_storeu_ps(st + i + 16, f2); _mm256_storeu_ps(st + i + 24, f3);
    }
    for (; i < I; i += 2) {
        uint8_t b = row[i >> 1];
        int8_t e0 = (int8_t)(b & 0xF);  if (e0 & 8)  e0 -= 16;
        int8_t e1 = (int8_t)(b >> 4);   if (e1 & 8)  e1 -= 16;
        float a = (float)e0, c = (float)e1;
        if (scale) { a *= rs; c *= rs; }
        st[i] = a; st[i+1] = c;
    }
}

__attribute__((target("avx2,fma")))
static inline float q36_m4_hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

/* One output row against N (1..4) activation rows. Weights are decoded ONCE
 * and reused for all N tokens (the speculative-verify case). Signed int4 x
 * int8 via |w| (unsigned) x sign-adjusted x: no saturation, since
 * 2 * 8 * 127 per pair, summed twice, stays < 32767. */
__attribute__((target("avx2,fma"), always_inline))
static inline void q36_m4_row_chunk(float *yout, int64_t ystride, const uint8_t *row, float rs,
                                    const int8_t *xe, const int8_t *xo, const float *sg,
                                    int ng, int s0, int N) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), eight = _mm256_set1_epi8(8), ones = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (int g = 0; g < ng; g++) {
        __m256i b  = _mm256_loadu_si256((const __m256i *)(row + (int64_t)g * 32));
        __m256i wl = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(b, m4), eight), eight);
        __m256i wh = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(b, 4), m4), eight), eight);
        __m256i al = _mm256_abs_epi8(wl), ah = _mm256_abs_epi8(wh);
#define Q36_M4_TOK(j, acc) do { \
            int64_t gi = (int64_t)(s0 + (j)) * ng + g; \
            __m256i xe_ = _mm256_loadu_si256((const __m256i *)(xe + gi * 32)); \
            __m256i xo_ = _mm256_loadu_si256((const __m256i *)(xo + gi * 32)); \
            __m256i t = _mm256_add_epi16(_mm256_maddubs_epi16(al, _mm256_sign_epi8(xe_, wl)), \
                                         _mm256_maddubs_epi16(ah, _mm256_sign_epi8(xo_, wh))); \
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(t, ones)), _mm256_set1_ps(sg[gi]), acc); \
        } while (0)
        Q36_M4_TOK(0, acc0);
        if (N > 1) Q36_M4_TOK(1, acc1);
        if (N > 2) Q36_M4_TOK(2, acc2);
        if (N > 3) Q36_M4_TOK(3, acc3);
#undef Q36_M4_TOK
    }
    yout[(int64_t)(s0    ) * ystride] = q36_m4_hsum256(acc0) * rs;
    if (N > 1) yout[(int64_t)(s0 + 1) * ystride] = q36_m4_hsum256(acc1) * rs;
    if (N > 2) yout[(int64_t)(s0 + 2) * ystride] = q36_m4_hsum256(acc2) * rs;
    if (N > 3) yout[(int64_t)(s0 + 3) * ystride] = q36_m4_hsum256(acc3) * rs;
}

__attribute__((target("avx2,fma")))
static void q36_m4_idot_avx2(float *y, const float *x, const uint8_t *W, const float *sc,
                             int S, int I, int O) {
    const int ng = I / 64;
    size_t nx = (size_t)S * ng * 32;
    int8_t *xe = (int8_t *)malloc(nx), *xo = (int8_t *)malloc(nx);
    float *sg = (float *)malloc((size_t)S * ng * sizeof(float));
    if (!xe || !xo || !sg) { fprintf(stderr, "OOM mapi4 idot\n"); exit(1); }
    /* activations -> int8, one scale per 64 elements, split into even/odd
     * elements so they line up with the low/high nibbles with no shuffle. */
    for (int s = 0; s < S; s++)
        for (int g = 0; g < ng; g++) {
            const float *xp = x + (int64_t)s * I + (int64_t)g * 64;
            float amax = 0.0f;
            for (int k = 0; k < 64; k++) { float a = fabsf(xp[k]); if (a > amax) amax = a; }
            float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            int64_t gi = (int64_t)s * ng + g;
            sg[gi] = amax > 0.0f ? amax / 127.0f : 0.0f;
            for (int k = 0; k < 32; k++) {
                int a = (int)lrintf(xp[2 * k] * inv), c = (int)lrintf(xp[2 * k + 1] * inv);
                xe[gi * 32 + k] = (int8_t)(a > 127 ? 127 : a < -127 ? -127 : a);
                xo[gi * 32 + k] = (int8_t)(c > 127 ? 127 : c < -127 ? -127 : c);
            }
        }
    #pragma omp parallel for schedule(static) if (O >= 32)
    for (int o = 0; o < O; o++) {
        const uint8_t *row = W + (int64_t)o * (I >> 1);
        float rs = sc ? sc[o] : 1.0f;
        float *yo = y + o;
        int s0 = 0;
        for (; s0 + 4 <= S; s0 += 4) q36_m4_row_chunk(yo, O, row, rs, xe, xo, sg, ng, s0, 4);
        switch (S - s0) {
            case 3: q36_m4_row_chunk(yo, O, row, rs, xe, xo, sg, ng, s0, 3); break;
            case 2: q36_m4_row_chunk(yo, O, row, rs, xe, xo, sg, ng, s0, 2); break;
            case 1: q36_m4_row_chunk(yo, O, row, rs, xe, xo, sg, ng, s0, 1); break;
            default: break;
        }
    }
    free(xe); free(xo); free(sg);
}
#endif /* Q36_M4_X86 */

/* mode: 0 = scalar (original), 1 = AVX2 bit-exact staging, 2 = AVX2 integer dot */
static int q36_m4_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        int m = 0;
#ifdef Q36_M4_X86
        const char *sca = getenv("Q36_MAPI4_SCALAR"), *idt = getenv("Q36_MAPI4_IDOT");
        __builtin_cpu_init();
        if (!(sca && sca[0] == '1') && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            m = 1;
            if (idt && idt[0] == '1') m = 2;
        }
#endif
        mode = m;
        fprintf(stderr, "[mapi4] %s\n", m == 2 ? "AVX2 integer-dot kernel (Q36_MAPI4_IDOT=1; NOT bit-exact)"
                                      : m == 1 ? "AVX2 staging kernel (bit-exact with the scalar loop)"
                                               : "scalar staging loop");
    }
    return mode;
}

static inline void q36_m4_stage_row(float *st, const uint8_t *row, int I, float rs) {
#ifdef Q36_M4_X86
    if (q36_m4_mode() >= 1) { q36_m4_stage_avx2(st, row, I, rs); return; }
#endif
    q36_m4_stage_scalar(st, row, I, rs);
}

/* 1 if the integer-dot path should handle this matrix, and runs it. */
static inline int q36_m4_try_idot(float *y, const float *x, const uint8_t *W, const float *sc,
                                  int S, int I, int O) {
#ifdef Q36_M4_X86
    if (q36_m4_mode() == 2 && (I % 64) == 0 && S >= 1) { q36_m4_idot_avx2(y, x, W, sc, S, I, O); return 1; }
#endif
    (void)y; (void)x; (void)W; (void)sc; (void)S; (void)I; (void)O;
    return 0;
}

#endif /* COLI_QWEN_MAPI4_AVX2_H */
