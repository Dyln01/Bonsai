/* kv8_parity.c — assert kv8_dec1() == coli_fp8_lut[] for all 256 byte codes,
 * and measure quant->decode round-trip error on random rows (the lossy-KV
 * quality bound for the qwen36 KV8 patch).
 * build: gcc -O2 -I. kv8_parity.c -o /tmp/kv8_parity -lm  (run from c/)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kv_fp8.h"
#include "kv8_fused.h"

int main(void){
    coli_fp8_lut_init();   /* MUST come before any LUT read */
    int bad = 0;
    for (int b = 0; b < 256; b++) {
        float a = kv8_dec1((uint8_t)b);
        float av = kv8_dec_vec((uint32_t)b);
        if (av != a && !(av == 0.f && a == 0.f)) { printf("VEC MISMATCH b=%d vec=%.9g dec1=%.9g\n", b, av, a); bad++; }
        float l = coli_fp8_lut[b];
        /* the only allowed difference: signed zero */
        if (a != l && !(a == 0.f && l == 0.f)) {
            printf("MISMATCH b=%3d dec1=%.9g lut=%.9g\n", b, a, l);
            bad++;
        }
    }
    if (bad) { printf("FAIL: %d mismatches\n", bad); return 1; }
    printf("parity: all 256 codes identical (signed-zero tolerant)\n");

    /* round-trip error bound on random rows */
    srand(1234);
    double max_rel = 0, sum_rel = 0; long n = 0;
    for (int trial = 0; trial < 2000; trial++) {
        float src[128];
        uint8_t q[128];
        int w = 64 + (trial % 65);
        for (int i = 0; i < w; i++)
            src[i] = ((float)rand() / RAND_MAX - 0.5f) * (float)(1 << (rand() % 8));
        float s = coli_kv8_quant_row(src, q, w);
        for (int i = 0; i < w; i++) {
            float d = coli_fp8_lut[q[i]] * s;   /* engine decoder; parity proven above */
            float denom = fabsf(src[i]) > 1e-30f ? fabsf(src[i]) : 1e-30f;
            double rel = fabs(d - src[i]) / denom;
            if (rel > max_rel) max_rel = rel;
            sum_rel += rel; n++;
        }
    }
    printf("round-trip over %ld elems: max rel err %.4g, mean %.4g\n", n, max_rel, sum_rel / n);
    /* e4m3: relative half-step 1/16 = 0.0625. Per-row amax scaling can also
     * flush near-zero values (rel err 1) — that is the codec design, not a
     * decoder bug, so the gate is on the MEAN. */
    if (sum_rel / n > 0.0625) { printf("FAIL: mean error above e4m3 half-step\n"); return 1; }
    printf("PASS\n");
    return 0;
}
