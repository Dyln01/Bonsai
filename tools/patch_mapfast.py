#!/usr/bin/env python3
"""patch_mapfast.py — make the mapped GEMVs compute-efficient (TRUNK_MAP v3.1).

Root cause of the 288 s prefill: matmul_map_f16/matmul_map_i4 dequantize each
weight per (output-row, input-row) pair — scalar bit-twiddling at S*O*I cost.
Fix, same tile-staging trick as the KV8 patch: dequantize each output row ONCE
into per-thread f32 staging, then run the engine's own vectorized fixed-lane dot
over the S input rows. Compute drops from S*O*I dequants to O*I + S*O*I FMAs.
"""
import os, sys

P = os.path.expanduser("~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
if "MAPFAST" in src:
    print("already patched"); sys.exit(0)

def sub1(anchor, repl, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag}: count={src.count(anchor)}"
    src = src.replace(anchor, repl)

# Replace both mapped kernels with staged versions. Anchor: start of matmul_map_f16
# through the end of matmul_map_i4 (they are contiguous: f16 kernel then i4 kernel then matmul_d).
A_START = "static void matmul_map_f16(float *y, const float *x, const QW *w, int S, int I, int O) {"
A_END = "static void matmul_d(float *y, const float *x, const QW *w, int S, int I, int O){"
i0 = src.find(A_START)
i1 = src.find(A_END)
assert 0 < i0 < i1, (i0, i1)

NEW = '''/* MAPFAST v3.1: stage one output row's dequantized f32 per thread, then the
 * same fixed-lane dot the resident kernels use, over all S input rows. */
static inline float map_f16_one(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu) << 23;
    uint32_t man  = (uint32_t)(h & 0x3FFu) << 13;
    uint32_t bits = sign | exp | man;
    if (((h >> 10) & 0x1F) == 0) bits = sign;   /* subnormal -> +-0 */
    float f; memcpy(&f, &bits, 4); return f;
}

static void matmul_map_f16(float *y, const float *x, const QW *w, int S, int I, int O) {
    const uint16_t *W = (const uint16_t *)w->map;
    #pragma omp parallel
    {
        float *st = malloc((size_t)I * sizeof(float));
        if (!st) { fprintf(stderr, "OOM map stage\\n"); exit(1); }
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint16_t *row = W + (int64_t)o * I;
            for (int i = 0; i < I; i++) st[i] = map_f16_one(row[i]);
            for (int s = 0; s < S; s++)
                y[(int64_t)s * O + o] = dot_f32_lanes(x + (int64_t)s * I, st, I);
        }
        free(st);
    }
}

static void matmul_map_i4(float *y, const float *x, const QW *w, int S, int I, int O) {
    const uint8_t *W = (const uint8_t *)w->map;
    const float *sc = w->sc;
    #pragma omp parallel
    {
        float *st = malloc((size_t)I * sizeof(float));
        if (!st) { fprintf(stderr, "OOM map stage\\n"); exit(1); }
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *row = W + (int64_t)o * (I >> 1);
            for (int i = 0; i < I; i += 2) {
                uint8_t b = row[i >> 1];
                int8_t e0 = (int8_t)(b & 0xF);  if (e0 & 8)  e0 -= 16;
                int8_t e1 = (int8_t)(b >> 4);   if (e1 & 8)  e1 -= 16;
                st[i] = (float)e0; st[i+1] = (float)e1;
            }
            float rs = sc ? sc[o] : 1.0f;
            if (rs != 1.0f) for (int i = 0; i < I; i++) st[i] *= rs;
            for (int s = 0; s < S; s++)
                y[(int64_t)s * O + o] = dot_f32_lanes(x + (int64_t)s * I, st, I);
        }
        free(st);
    }
}

'''
src = src[:i0] + NEW + src[i1:]
open(P, "w").write(src)
print("MAPFAST v3.1 applied")
