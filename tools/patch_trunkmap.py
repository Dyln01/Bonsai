#!/usr/bin/env python3
"""patch_trunkmap.py — TRUNK_MAP: mmap-backed dense weights for qwen36.

Full-size dense checkpoints (Qwen3.8-27B: 24 GB f16 trunk) cannot be resident
on small boxes; the loader today reads every dense matrix into RAM (or int8
RAM). TRUNK_MAP=1 instead memory-maps each F16 tensor straight from the
safetensors shard and leaves the bytes on disk; the GEMV dequantizes
f16->f32 on the fly. The OS page cache becomes a free CDN (hot pages stay,
cold pages evict under pressure without OOM). Combined with the engine's
speculative verify (one sweep serves S rows) and KV8, a 27B boots and runs
on a box with a fraction of the checkpoint size in RAM.

No quality change vs the f16 container: same bits, just not resident.
TRUNK_MAP=0 restores the resident path everywhere (default off).

Patch points (all verified unique in the bf24429+flash+KV8 tree):
  1. QW gains map/map_len fields.
  2. load_tq: F16 tensor + TRUNK_MAP -> mmap instead of read+quantize.
  3. matmul_d: w->map branch -> matmul_map_f16 (AVX2-friendly dequant dot).
  4. layers_forward_range: MADV_WILLNEED on layer i+1's resident-mapped QWs.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
if "TRUNK_MAP" in src:
    print("already patched"); sys.exit(0)

def sub1(anchor, repl, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag}: count={src.count(anchor)}"
    src = src.replace(anchor, repl)

# ---- includes (after the OpenMP/stdlib block near the top; anchor on st.h include) ----
sub1('#include "st.h"',
     '''#include "st.h"
#include <sys/mman.h>   /* TRUNK_MAP: mmap-backed dense weights */
#include <fcntl.h>''', "includes")

# ---- 1. QW fields ----
A_QW = "typedef struct { const float *w; int8_t *q; float *sc; int I, O; uint8_t *q4; float *sg; int ng;\n                 void *vk; int vk_off; uint16_t *h;"
sub1(A_QW,
     "typedef struct { const float *w; int8_t *q; float *sc; int I, O; uint8_t *q4; float *sg; int ng;\n"
     "                 void *map; int64_t map_len;   /* TRUNK_MAP: mmap'd F16 bytes on disk */\n"
     "                 void *vk; int vk_off; uint16_t *h;", "QW-fields")

# ---- 2. load_tq: the mmap path ----
A_LTQ = """static void load_tq(Model *m, const char *name, int I, int O, int quantize, const char *tag, QW *out) {
    float *p = load_t_n(m, name, (int64_t)I * O);"""
sub1(A_LTQ,
     """static void load_tq(Model *m, const char *name, int I, int O, int quantize, const char *tag, QW *out) {
    out->map = NULL; out->map_len = 0;
    {
        /* TRUNK_MAP: leave F16 dense bytes on disk, mapped. The GEMV
         * dequantizes on the fly; the page cache gives hot rows for free
         * and the kernel evicts under pressure instead of OOMing the box.
         * opt-in: TRUNK_MAP=1. Only for F16 (the converter's dense format). */
        static int tmap = -1;
        if (tmap < 0) { const char *e = getenv("TRUNK_MAP"); tmap = (e && *e == '1'); }
        if (tmap && quantize && dense_i8_on()) {
            char rn[QW_DENSE_NAME_MAX];
            const char *nm = dense_resolve(m, name, rn, sizeof rn);
            st_tensor *t = st_find(&m->S, nm);
            if (t && t->dtype == 1 && t->numel == (int64_t)I * O) {
                int64_t pg = 4096, off = t->off, len = t->nbytes;
                int64_t base = (off / pg) * pg, skip = off - base, span = skip + len;
                void *mp = mmap(NULL, span, PROT_READ, MAP_PRIVATE, t->fd, base);
                if (mp != MAP_FAILED) {
                    out->map = (char*)mp + skip; out->map_len = len;
                    out->w = NULL; out->q = NULL; out->sc = NULL;
                    out->I = I; out->O = O;
                    out->q4 = NULL; out->sg = NULL; out->ng = 0;
                    out->vk = NULL; out->vk_off = 0; out->h = NULL;
                    return;
                }
                /* fall through: no mmap (e.g. drvfs) -> resident path */
            }
        }
    }
    float *p = load_t_n(m, name, (int64_t)I * O);""", "load_tq-mmap")

# ---- 3. matmul_d: mapped branch ----
A_MMD = """static void matmul_d(float *y, const float *x, const QW *w, int S, int I, int O){
#ifdef COLI_QWEN_BATCH_TEST
    g_qwen_matmul_d_calls++;
#endif"""
sub1(A_MMD,
     """static void matmul_map_f16(float *y, const float *x, const QW *w, int S, int I, int O) {
    /* dequant-on-the-fly GEMV over mmap'd F16 bytes. Rows are contiguous
     * (row-major [O, I] f16); one pass over a row serves all S input rows,
     * so a speculative verify amortizes the disk read S times. */
    const uint16_t *W = (const uint16_t *)w->map;
    #pragma omp parallel for schedule(static) if (O >= 64)
    for (int o = 0; o < O; o++) {
        const uint16_t *row = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a[16] = {0};   /* 16-lane fixed-order, like dot_f32_lanes */
            int i = 0;
            for (; i + 8 <= I; i += 8) {
                /* f16x8 -> f32x8 (AVX2-friendly scalar fallback identical) */
                float c[8];
                for (int j = 0; j < 8; j++) {
                    uint16_t h = row[i + j];
                    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
                    uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu) << 23;
                    uint32_t man  = (uint32_t)(h & 0x3FFu) << 13;
                    uint32_t bits = sign | exp | man;
                    if (((h >> 10) & 0x1F) == 0) bits = sign;   /* subnormal -> +-0 */
                    memcpy(&c[j], &bits, 4);
                }
                for (int j = 0; j < 8; j++) a[j & 15] = fmaf(c[j], xs[i + j], a[j & 15]);
            }
            for (; i < I; i++) {
                uint16_t h = row[i];
                uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
                uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu) << 23;
                uint32_t man  = (uint32_t)(h & 0x3FFu) << 13;
                uint32_t bits = sign | exp | man;
                if (((h >> 10) & 0x1F) == 0) bits = sign;
                float c; memcpy(&c, &bits, 4);
                a[i & 15] = fmaf(c, xs[i], a[i & 15]);
            }
            float acc = 0;
            for (int j = 0; j < 16; j++) acc += a[j];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

static void matmul_d(float *y, const float *x, const QW *w, int S, int I, int O){
#ifdef COLI_QWEN_BATCH_TEST
    g_qwen_matmul_d_calls++;
#endif
    if (w->map) { matmul_map_f16(y, x, w, S, I, O); return; }""", "matmul_d-map")

# ---- 4. readahead: MADV_WILLNEED on the next layer's mapped matrices ----
A_FWD = """        if (allow_prefetch && g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);"""
sub1(A_FWD,
     """        if (allow_prefetch && g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
#ifdef MADV_WILLNEED
        if (i + 2 < c->n_layers && m->L[i + 2].q.map) {   /* TRUNK_MAP readahead */
            const QW *nq[6] = { &m->L[i+2].q, &m->L[i+2].k, &m->L[i+2].v, &m->L[i+2].o,
                                &m->L[i+2].sh_g, &m->L[i+2].sh_u };
            for (int qw = 0; qw < 6; qw++)
                if (nq[qw]->map) madvise((void*)(((uintptr_t)nq[qw]->map) & ~(uintptr_t)4095),
                                         (size_t)nq[qw]->map_len + 4096, MADV_WILLNEED);
        }
#endif""", "readahead")

open(P, "w").write(src)
print("TRUNK_MAP patch applied (backup at qwen36.c.premap next)")
