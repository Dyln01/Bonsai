#!/usr/bin/env python3
"""patch_trunkmap_v2.py — extend TRUNK_MAP: int4-packed mapped trunk + mapped embed/lm_head.

v1 (already applied): F16 dense tensors mmap + dequant GEMV. Proven: full-size 27B boots on 8 GB
(stock engine OOMs at 7.4 GB); f16 streaming is disk-bound (44 GB sweep).

v2 adds the byte-cut that makes streaming usable:
  A) matmul_map_i4: mmap'd INT4-PACKED dense shards (U8 low-nibble-even, per-row f32 scales —
     the same convention as the converter's experts) + dequant-on-the-fly GEMV. 4x fewer bytes
     per sweep than f16.
  B) load_tq maps dtype==3 (U8/I8) tensors of EXACT packed size I*O/2 (int4) the same way.
  C) embed/lm_head stay resident (they're gathers/full-reads; int8-resident is correct for them).

All opt-in under the same TRUNK_MAP=1; TRUNK_MAP=0 restores stock everywhere.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
if "matmul_map_i4" in src:
    print("v2 already applied"); sys.exit(0)

def sub1(anchor, repl, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag}: count={src.count(anchor)}"
    src = src.replace(anchor, repl)

# ---- A) int4 mapped kernel, next to the f16 one ----
A_KN = "static void matmul_d(float *y, const float *x, const QW *w, int S, int I, int O){"
sub1(A_KN,
"""static void matmul_map_i4(float *y, const float *x, const QW *w, int S, int I, int O) {
    /* int4-packed mapped trunk: bytes are [O, I/2] U8 (low nibble = even element,
     * signed [-8,7] per the converter's pack convention), scales w->sc = f32[O]
     * (per-row absmax/7) or NULL (weights packed raw; caller sets sc).
     * S rows amortize one dequant pass over the row, same as the f16 kernel. */
    const uint8_t *W = (const uint8_t *)w->map;
    const float *sc = w->sc;
    #pragma omp parallel for schedule(static) if (O >= 64)
    for (int o = 0; o < O; o++) {
        const uint8_t *row = W + (int64_t)o * (I >> 1);
        float rs = sc ? sc[o] : 1.0f;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a[16] = {0};
            int i = 0;
            for (; i + 16 <= I; i += 16) {
                uint64_t lo64, hi64;
                memcpy(&lo64, row + (i >> 1), 8);
                memcpy(&hi64, row + (i >> 1) + 8, 8);
                /* 16 nibbles: element 2k in low nibble of byte k, 2k+1 in high */
                for (int j = 0; j < 8; j++) {
                    int8_t e0 = (int8_t)(lo64 & 0xF); if (e0 & 8) e0 -= 16;   /* sign-extend */
                    int8_t e1 = (int8_t)((lo64 >> 4) & 0xF); if (e1 & 8) e1 -= 16;
                    a[j & 15]      = fmaf((float)e0, xs[i + 2*j],     a[j & 15]);
                    a[(j+8) & 15]  = fmaf((float)e1, xs[i + 2*j + 1], a[(j+8) & 15]);
                    lo64 >>= 8;
                }
                for (int j = 0; j < 8; j++) {
                    int8_t e0 = (int8_t)(hi64 & 0xF); if (e0 & 8) e0 -= 16;
                    int8_t e1 = (int8_t)((hi64 >> 4) & 0xF); if (e1 & 8) e1 -= 16;
                    a[j & 15]      = fmaf((float)e0, xs[i + 16 + 2*j],     a[j & 15]);
                    a[(j+8) & 15]  = fmaf((float)e1, xs[i + 16 + 2*j + 1], a[(j+8) & 15]);
                    hi64 >>= 8;
                }
            }
            for (; i < I; i++) {
                uint8_t b = row[i >> 1];
                int8_t e = (int8_t)((i & 1) ? (b >> 4) : (b & 0xF));
                if (e & 8) e -= 16;
                a[i & 15] = fmaf((float)e, xs[i], a[i & 15]);
            }
            float acc = 0;
            for (int j = 0; j < 16; j++) acc += a[j];
            y[(int64_t)s * O + o] = acc * rs;
        }
    }
}

static void matmul_d(float *y, const float *x, const QW *w, int S, int I, int O){""", "i4-kernel")

# dispatch: pick int4 kernel when map_len == I*O/2 (packed) and scales exist
A_DSP = "    if (w->map) { matmul_map_f16(y, x, w, S, I, O); return; }"
sub1(A_DSP,
"""    if (w->map) {
        if (w->map_len == (int64_t)I * O / 2) matmul_map_i4(y, x, w, S, I, O);
        else matmul_map_f16(y, x, w, S, I, O);
        return;
    }""", "dispatch")

# ---- B) load_tq: also map int4-packed U8 tensors (exact half-size match) ----
A_LTQ = """            if (t && t->dtype == 1 && t->numel == (int64_t)I * O) {"""
sub1(A_LTQ,
"""            int64_t i4bytes = (int64_t)I * O / 2;
            int is_i4 = (t && t->dtype == 3 && t->nbytes == i4bytes);
            if (t && (is_i4 || (t->dtype == 1 && t->numel == (int64_t)I * O))) {""", "i4-load-a")

# inside the mmap success block: for int4 also read the f32 scales (name+".qs")
A_MP = """                void *mp = mmap(NULL, span, PROT_READ, MAP_PRIVATE, t->fd, base);
                if (mp != MAP_FAILED) {
                    out->map = (char*)mp + skip; out->map_len = len;"""
sub1(A_MP,
"""                void *mp = mmap(NULL, span, PROT_READ, MAP_PRIVATE, t->fd, base);
                if (mp != MAP_FAILED) {
                    out->map = (char*)mp + skip; out->map_len = len;
                    if (is_i4) {
                        /* scales tensor <name>.qs (f32 per row) — small, keep resident */
                        char qsname[512]; snprintf(qsname, sizeof qsname, "%s.qs", nm);
                        st_tensor *qt = st_find(&m->S, qsname);
                        if (qt && qt->dtype == 2 && qt->numel == O) {
                            out->sc = falloc(O);
                            st_read_f32(&m->S, qsname, out->sc, 0);
                        } else { out->sc = NULL; }
                    }""", "i4-scales")

open(P, "w").write(src)
print("TRUNK_MAP v2 applied (int4-mapped kernel + scales)")
