#!/usr/bin/env python3
"""patch_softmax.py — Q36_FAST_SOFTMAX=1: AVX2 fast exponential in qwen36's
attention softmax (+ the flash prefill's two online-softmax expf sites).

Stock softmax_row is three passes over the O(T) score row with a libm expf
call per element; nothing vectorizes it without -ffast-math. The fast path
replaces exp with an exponent-bit-trick 2^x (degree-5 minimax poly fitted on
[-0.5,0.5], two-FMA exact argument reduction), vectorizes all three passes
8-wide, and multiplies by 1/s instead of dividing.

Gated by Q36_FAST_SOFTMAX=1 (opt-in, default stock byte-identical), and only
when the build has AVX2+FMA (the engine's default -march=x86-64-v3 provides
them; on arm64/other builds the knob prints a warning and stays off).

Correctness (softmaxbench_q36.c, engine/ in this repo):
  - max rel err vs expf over [-104,0]: 2.46e-07 (at x=-11.4)
  - underflow zone [-104,-87): flushes to exact 0 (the exponent field must
    never wrap: n+127 >= 1) — a N(0,15) gate arm exists because this exact
    bug shipped in the bench's first version and produced NaN
  - softmax output vs an f64 reference: the fast path is MORE accurate than
    stock (5.9e-8 .. 9.6e-8 worst elementwise vs stock's 3.0e-7 .. 1.3e-6;
    stock's plain f32 accumulator drifts ~1.3e-5 on the row sum at 16k)
Measured (Ryzen 7 7735HS, -O3 -march=x86-64-v3): 9.5x-11.5x on rows of
1024..32768; scalar-only fast exp is 1.07x, so the win IS the vectorization.
Lossy-by-rounding label: a 2.5e-7 exp perturbation can flip a near-tie
greedy token, same class as the Vulkan-tier note in docs/speculative.md.

NEGATIVE RESULT, kept so nobody re-tries it blind: an earlier version of
this patch ALSO swapped the flash prefill's two online-softmax expf sites to
the fast exp. That cost +13% TTFT in 4/4 runs (169.8-174.8s vs 148.6-156.1s
stock; an order-swapped control ruled out arm-position effects) even though
the flash loop performs only ~2.1M exps (~10 ms) per prefill — a codegen
perturbation of the surrounding V-accumulate loop ~2000x larger than the
exp work itself. Reverting just those two sites (isolation build) restored
TTFT to 143.4s while keeping the full decode win. The shipped knob covers
softmax_row ONLY; leave the flash exps alone.

Anchored byte-exact edits (sub1 count==1 asserts), backup qwen36.c.presm.
Refuses to run twice. Usage: python3 patch_softmax.py <path-to-qwen36.c>
"""
import os, shutil, sys

EDITS = []
def edit(old, new): EDITS.append((old, new))

# ---------------------------------------------------------------- E1: fast-exp block + dispatcher
edit(
'static void softmax_row(float *x, int n) {\n'
'    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];\n'
'    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }\n'
'    for (int i = 0; i < n; i++) x[i] /= s;\n'
'}\n',
'/* ---- Q36_FAST_SOFTMAX=1: AVX2 fast exponential (opt-in, lossy-by-rounding) ----\n'
' *\n'
' * 2^x = 2^n * poly(f) with n = round(x*log2e), f in [-0.5,0.5]: the integer\n'
' * part goes straight into the exponent field, the fraction through a\n'
' * degree-5 minimax polynomial (Chebyshev-node fit on [-0.5,0.5]; the\n'
' * coefficients are measured, not copied from memory). The argument is\n'
' * reduced with two FMAs (L_HI + L_LO split of log2e) so the fraction never\n'
' * inherits the |x*log2e| rounding error that would otherwise grow linearly\n'
' * with |x|. x < -87 flushes to 0: below that the exponent field would wrap\n'
' * (n+127 < 1) and the bit trick produces garbage — softmaxbench_q36.c gates\n'
' * that zone explicitly, because its first version shipped exactly that bug.\n'
' *\n'
' * softmaxbench_q36.c (engine/) gates the whole thing:\n'
' *   max rel err vs expf over [-104,0]: 2.46e-07\n'
' *   softmax vs an f64 reference: MORE accurate than the stock 3-pass\n'
' *   (stock\'s plain f32 accumulator drifts ~1.3e-5 on the 16k row sum)\n'
' * Measured: 9.5x-11.5x per row (1024..32768) on x86-64-v3. The win is the\n'
' * vectorization: scalar fast exp is only 1.07x over glibc expf.\n'
' *\n'
' * Lossy-by-rounding: a 2.5e-7 exp perturbation can flip a near-tie greedy\n'
' * token (the Vulkan-tier note in docs/speculative.md describes the same\n'
' * class). Opt-in for that reason; the default path is stock verbatim. */\n'
'static int g_q36_fast_sm = -1;   /* -1 unparsed, 0 off, 1 on */\n'
'static int q36_fast_sm_on(void) {\n'
'    if (g_q36_fast_sm < 0) {\n'
'        const char *e = getenv("Q36_FAST_SOFTMAX");\n'
'        g_q36_fast_sm = (e && *e == \'1\') ? 1 : 0;\n'
'#if !(defined(__AVX2__) && defined(__FMA__))\n'
'        if (g_q36_fast_sm) {\n'
'            fprintf(stderr, "[qwen36] Q36_FAST_SOFTMAX=1 ignored: this build has no AVX2+FMA\\n");\n'
'            g_q36_fast_sm = 0;\n'
'        }\n'
'#endif\n'
'    }\n'
'    return g_q36_fast_sm;\n'
'}\n'
'\n'
'static inline float q36_fexp(float x) {\n'
'    if (x < -87.0f) return 0.f;                 /* exponent field must never wrap */\n'
'    const float L_HI = 1.4426950216293335f;      /* f32-nearest log2(e) */\n'
'    const float L_LO = 1.925963e-8f;            /* log2(e) - L_HI */\n'
'    float t = x * L_HI;                          /* coarse: only to pick n */\n'
'    float n = (float)(int)(t + (t < 0.f ? -0.5f : 0.5f));\n'
'    float f = fmaf(x, L_LO, fmaf(x, L_HI, -n));  /* exact fraction: 2 FMAs */\n'
'    float p = 1.0000001f + f * (0.6931472f + f * (0.24022107f +\n'
'              f * (0.05550327f + f * (0.009676037f + f * 0.0013400433f))));\n'
'    union { uint32_t i; float f; } u;\n'
'    u.i = (uint32_t)((int)n + 127) << 23;\n'
'    return p * u.f;\n'
'}\n'
'\n'
'/* exp that follows the knob: used by the flash prefill\'s online softmax. */\n'
'static inline float q36_exp(float x) { return q36_fast_sm_on() ? q36_fexp(x) : expf(x); }\n'
'\n'
'#if defined(__AVX2__) && defined(__FMA__)\n'
'static inline __m256 q36_fexp8(__m256 x) {\n'
'    const __m256 FLOOR = _mm256_set1_ps(-87.0f);\n'
'    const __m256 L_HI = _mm256_set1_ps(1.4426950216293335f), L_LO = _mm256_set1_ps(1.925963e-8f);\n'
'    const __m256 C0 = _mm256_set1_ps(1.0000001f), C1 = _mm256_set1_ps(0.6931472f);\n'
'    const __m256 C2 = _mm256_set1_ps(0.24022107f), C3 = _mm256_set1_ps(0.05550327f);\n'
'    const __m256 C4 = _mm256_set1_ps(0.009676037f), C5 = _mm256_set1_ps(0.0013400433f);\n'
'    const __m256i BIAS = _mm256_set1_epi32(127);\n'
'    __m256 mask = _mm256_cmp_ps(x, FLOOR, _CMP_LT_OQ);\n'
'    x = _mm256_max_ps(x, FLOOR);\n'
'    __m256 t = _mm256_mul_ps(x, L_HI);\n'
'    __m256 rnd = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);\n'
'    __m256 f = _mm256_fmadd_ps(x, L_LO, _mm256_fmsub_ps(x, L_HI, rnd));\n'
'    __m256 p = C5;\n'
'    p = _mm256_fmadd_ps(f, p, C4);\n'
'    p = _mm256_fmadd_ps(f, p, C3);\n'
'    p = _mm256_fmadd_ps(f, p, C2);\n'
'    p = _mm256_fmadd_ps(f, p, C1);\n'
'    p = _mm256_fmadd_ps(f, p, C0);\n'
'    __m256i e = _mm256_add_epi32(_mm256_cvtps_epi32(rnd), BIAS);\n'
'    __m256 two_n = _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));\n'
'    return _mm256_andnot_ps(mask, _mm256_mul_ps(p, two_n));\n'
'}\n'
'\n'
'static void softmax_row_fast(float *x, int n) {\n'
'    __m256 vm = _mm256_set1_ps(-1e30f);\n'
'    int i = 0;\n'
'    for (; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(x + i));\n'
'    __m128 lo = _mm256_castps256_ps128(vm), hi = _mm256_extractf128_ps(vm, 1);\n'
'    lo = _mm_max_ps(lo, hi);\n'
'    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));\n'
'    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 1));\n'
'    float m = _mm_cvtss_f32(lo);\n'
'    for (; i < n; i++) if (x[i] > m) m = x[i];\n'
'    __m256 vs = _mm256_setzero_ps();\n'
'    i = 0;\n'
'    for (; i + 8 <= n; i += 8) {\n'
'        __m256 e = q36_fexp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(m)));\n'
'        _mm256_storeu_ps(x + i, e);\n'
'        vs = _mm256_add_ps(vs, e);\n'
'    }\n'
'    __m128 slo = _mm256_castps256_ps128(vs), shi = _mm256_extractf128_ps(vs, 1);\n'
'    __m128 sd = _mm_add_ps(slo, shi);\n'
'    sd = _mm_add_ps(sd, _mm_movehl_ps(sd, sd));\n'
'    sd = _mm_add_ss(sd, _mm_shuffle_ps(sd, sd, 1));\n'
'    float s = _mm_cvtss_f32(sd);\n'
'    for (; i < n; i++) { float e = q36_fexp(x[i]-m); x[i] = e; s += e; }\n'
'    float inv = 1.f / s;\n'
'    for (i = 0; i + 8 <= n; i += 8)\n'
'        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(inv)));\n'
'    for (; i < n; i++) x[i] *= inv;\n'
'}\n'
'#endif\n'
'\n'
'static void softmax_row(float *x, int n) {\n'
'#if defined(__AVX2__) && defined(__FMA__)\n'
'    if (q36_fast_sm_on()) { softmax_row_fast(x, n); return; }\n'
'#endif\n'
'    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];\n'
'    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }\n'
'    for (int i = 0; i < n; i++) x[i] /= s;\n'
'}\n'
)

# ---------------------------------------------------------------- E4: startup banner
edit(
'    if (m.kv8) coli_fp8_lut_init();\n'
'    if (m.kv8) fprintf(stderr, "[qwen36] KV8=1: KV rows stored fp8-e4m3 + per-row f32 scale (lossy, opt-in)\\n");\n',
'    if (m.kv8) coli_fp8_lut_init();\n'
'    if (m.kv8) fprintf(stderr, "[qwen36] KV8=1: KV rows stored fp8-e4m3 + per-row f32 scale (lossy, opt-in)\\n");\n'
'    q36_fast_sm_on();   /* Q36_FAST_SOFTMAX: parse once, warn once */\n'
'    if (g_q36_fast_sm == 1)\n'
'        fprintf(stderr, "[qwen36] Q36_FAST_SOFTMAX=1: AVX2 fast exp in the attention softmax "\n'
'                "(max rel err 2.5e-7 vs expf; lossy-by-rounding, opt-in)\\n");\n'
)

def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
    src = open(path, "r", encoding="utf-8", newline="").read()
    if "Q36_FAST_SOFTMAX" in src:
        sys.exit("refusing: qwen36.c already carries the fast-softmax markers (idempotence guard)")
    for i, (old, new) in enumerate(EDITS):
        n = src.count(old)
        if n != 1:
            sys.exit("E%02d: anchor count=%d (need 1): %.80r" % (i + 1, n, old))
        src = src.replace(old, new, 1)
    bak = path + ".presm"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print("backup -> %s" % bak)
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    print("applied %d edits -> %s" % (len(EDITS), path))

if __name__ == "__main__":
    main()
