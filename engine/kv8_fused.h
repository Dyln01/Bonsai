/* kv8_fused.h — v3: branch-free e4m3 decode + tile-staged dequant.
 *
 * v1 (LUT in the dot): 1.7x slower than f32. v2 (bit-math, branchy): worse.
 * v3 decodes TILES of KV rows into a per-thread f32 staging buffer with a
 * branch-free conversion, then runs the SAME dot_f32_lanes the f32 path uses,
 * so the hot loop is the vectorized kernel. Tile size = 256 rows: 256*kvd*4B
 * = 64 KiB at kvd=64 (fits L2; the dot re-reads it H/q_per_kv times per tile).
 *
 * Branch-free decode, all 256 codes, value-identical to coli_fp8_lut
 * (signed-zero tolerant; asserted by kv8_parity.c):
 *   bits = sign | ((e+120)<<23) | (m<<20)      // normal map
 *   den = (bits - 2^-7) * 2                    // exact when e==0
 *   val = e==0 ? den : bits ; NaNcode -> 0
 * done with float select (bitwise AND on the exponent) instead of branches:
 *   mask = (e==0) ? 0xFFFFFFFF : 0 ; v = bits*mask... — simpler: compute
 *   both and blend by integer mask, still branch-free and vectorizes.
 */
#ifndef COLI_QWEN_KV8_FUSED_H
#define COLI_QWEN_KV8_FUSED_H

#include "kv_fp8.h"
#include "matmul_f32.h"

static inline float kv8_dec1(uint8_t b){
    uint32_t u = b;
    uint32_t e = (u >> 3) & 0xF;
    uint32_t m = u & 7;
    uint32_t bits = ((e + 120) << 23) | (m << 20);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    if (e == 0) f = (f - 0x1p-7f) * 2.0f;
    if ((u & 0x7F) == 0x7F) f = 0.0f;
    return (u & 0x80u) ? -f : f;
}

/* branch-free variant used in the tile loop (vectorizes) */
static inline float kv8_dec_vec(uint32_t u){
    uint32_t e = (u >> 3) & 0xF;
    uint32_t bits = ((e + 120) << 23) | ((u & 7) << 20);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    float den = (f - 0x1p-7f) * 2.0f;              /* value if e==0 */
    /* select: e==0 -> den, else f. NaN code (0x7F): f is huge; zero it. */
    uint32_t is_den = 0u - (uint32_t)(e == 0);          /* all-ones when e==0 */
    uint32_t isnan = 0u - (uint32_t)((u & 0x7F) == 0x7F);
    uint32_t fb, db;
    __builtin_memcpy(&fb, &f, 4);
    __builtin_memcpy(&db, &den, 4);
    uint32_t out = (fb & ~is_den) | (db & is_den);
    float g;
    __builtin_memcpy(&g, &out, 4);
    /* NaN code: zero. f for 0x7F would be 0x7F800000-ish; blend to 0. */
    uint32_t zero = 0;
    uint32_t out2 = (out & ~isnan) | (zero & isnan);
    float h;
    __builtin_memcpy(&h, &out2, 4);
    return (u & 0x80u) ? -h : h;
}

/* dequant WITHOUT the row scale (the caller applies it once per row) */
static inline void kv8_dequant_noscale(const uint8_t *src, float *dst, int n){
    for (int i = 0; i < n; i++) dst[i] = kv8_dec_vec(src[i]);
}

/* dequant one KV row (kvd floats) into dst — no branches in the loop body */
static inline void kv8_dequant_row_fast(const uint8_t *src, float scale, float *dst, int n){
    for (int i = 0; i < n; i++) dst[i] = kv8_dec_vec(src[i]) * scale;
}

#endif
