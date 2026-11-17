#!/usr/bin/env python3
"""patch_down_compact.py — Q36_DOWN_COMPACT=1: keep only the HOT columns of each layer's down_proj.

Field logs (bench_20261117 mem_fix runs): even with the embedding off RAM the box re-reads ~7.5 GB per
forward pass, against a 5.0 GB page cache. The cost model predicts 7.68 GB at frac 0.25 once the
down_proj layout is accounted for — within 1-3% of the measurement. down_proj is stored [hidden, inter]
row-major, so the hot neurons are the first Ik COLUMNS of each of 5120 rows: nearly every 4 KB page of the
matrix is touched (71% of it at frac 0.25, never less than ~1.3 GB over 64 layers however small frac is),
instead of the 0.71 GB the hot bytes occupy.

This patch copies each layer's hot columns once, on first use, into a compact buffer (rows Ik/2 bytes apart
instead of I/2) and reads that from then on; the strided mapped pages are never touched again and become
plain reclaimable page cache (MADV_DONTNEED). Working set for down: ~0.71 GB at frac 0.25 (anonymous),
not ~2.0 GB of page cache. No container change, no re-pack.

BIT-EXACT: both consumers (q36_down_prefix for decode, q36_down_prefix_batch for prefill/verify) run the
same loops over the same bytes; only the row pointer / stride differ. OFF by default.

Two anchored edits per function, house style. Backup qwen36.c.predcmp.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "Q36_DOWN_COMPACT" in src:
    raise SystemExit("patch_down_compact: already applied")

HELPER = r'''/* ---- Q36_DOWN_COMPACT=1: the hot columns of each layer's mapped int4 down_proj, compacted ----
 * down is [D rows][I cols]; the hot neurons are columns [0,Ik) of EVERY row, so reading them through the
 * mapping touches ~every page of the matrix. Copy them once into [D rows][(Ik+1)/2 bytes] and read that. */
typedef struct { const void *key; int Ik; uint8_t *buf; int64_t stride; } Q36DCompact;
static Q36DCompact g_q36_dc[256];
static int g_q36_dc_n = 0, g_q36_dc_on = -1;
static double g_q36_dc_mb = 0.0;
static void q36_dc_report(void) {
    fprintf(stderr, "[down-compact] %d layer(s), %.0f MB of hot down_proj columns held compact (anonymous)\n",
            g_q36_dc_n, g_q36_dc_mb);
}
static int q36_dc_on(void) {
    if (g_q36_dc_on < 0) {
        const char *e = getenv("Q36_DOWN_COMPACT");
        g_q36_dc_on = (e && e[0] == '1');
        if (g_q36_dc_on) atexit(q36_dc_report);
    }
    return g_q36_dc_on;
}
/* Row pointer + byte stride for the mapped int4 down matrix: the original mapping (stride I/2) when off,
 * else the compact copy (stride (Ik+1)/2). Called outside any parallel region (first use copies).
 * Keyed by (mapping address, Ik): safe because this engine never munmaps a matrix, so an address is
 * never reused for different bytes. */
static const uint8_t *q36_dc_rows(const QW *dw, int D, int I, int Ik, int64_t *stride) {
    const uint8_t *mp = (const uint8_t *)dw->map;
    *stride = I >> 1;
    if (!q36_dc_on()) return mp;
    for (int k = 0; k < g_q36_dc_n; k++)
        if (g_q36_dc[k].key == dw->map && g_q36_dc[k].Ik == Ik) { *stride = g_q36_dc[k].stride; return g_q36_dc[k].buf; }
    if (g_q36_dc_n >= 256) return mp;                      /* table full: still correct, just not compact */
    int64_t cb = (Ik + 1) >> 1;
    uint8_t *buf = (uint8_t *)malloc((size_t)D * (size_t)cb);
    if (!buf) { fprintf(stderr, "[down-compact] out of memory; this layer stays strided\n"); return mp; }
    int64_t full = (int64_t)D * (I >> 1);
    uintptr_t a0 = (uintptr_t)mp & ~(uintptr_t)4095, a1 = (uintptr_t)mp + (uintptr_t)full;
    madvise((void *)a0, (size_t)(a1 - a0), MADV_SEQUENTIAL);   /* one forward sweep: let readahead work */
    for (int d = 0; d < D; d++) memcpy(buf + (int64_t)d * cb, mp + (int64_t)d * (I >> 1), (size_t)cb);
    madvise((void *)a0, (size_t)(a1 - a0), MADV_DONTNEED);     /* read-only mapping: drop it, never touched again */
    g_q36_dc[g_q36_dc_n].key = dw->map; g_q36_dc[g_q36_dc_n].Ik = Ik;
    g_q36_dc[g_q36_dc_n].buf = buf;     g_q36_dc[g_q36_dc_n].stride = cb;
    g_q36_dc_n++;
    g_q36_dc_mb += (double)D * (double)cb / 1048576.0;
    *stride = cb;
    return buf;
}

'''

BATCH_SIG = "static void q36_down_prefix_batch(const QW *dw, float *dst, const float *g,\n"
SINGLE_SIG = "static void q36_down_prefix(const QW *dw, float *dst, const float *g, int D, int I, int Ik) {\n"
MP_OLD = "const uint8_t *mp = (const uint8_t*)dw->map; const float *sc = dw->sc;\n"
MP_NEW = ("int64_t dstride; const uint8_t *mp = q36_dc_rows(dw, D, I, Ik, &dstride); const float *sc = dw->sc;\n")
ROW_OLD = "const uint8_t *row = mp + (int64_t)d * (I >> 1);\n"
ROW_NEW = "const uint8_t *row = mp + (int64_t)d * dstride;\n"

if src.count(BATCH_SIG) != 1 or src.count(SINGLE_SIG) != 1:
    raise SystemExit("patch_down_compact: function anchors not found exactly once — wrong qwen36.c?")

# the two functions are adjacent: batch first, then single, then the Q36Win block
a = src.index(BATCH_SIG)
b = src.index(SINGLE_SIG)
c = src.index("/* ---- Phase 4b: resident neuron-row window", b)
if not (a < b < c):
    raise SystemExit("patch_down_compact: unexpected function order")

def edit(chunk, name):
    for old, new in ((MP_OLD, MP_NEW), (ROW_OLD, ROW_NEW)):
        n = chunk.count(old)
        if n != 1:
            raise SystemExit(f"patch_down_compact: {name}: anchor {old.strip()[:50]!r} found {n}x (want 1)")
        chunk = chunk.replace(old, new, 1)
    return chunk

batch = edit(src[a:b], "q36_down_prefix_batch")
single = edit(src[b:c], "q36_down_prefix")
out = src[:a] + HELPER + batch + single + src[c:]
open(P, "wb").write(out.encode("utf8", "surrogateescape"))
print("patch_down_compact: applied (Q36_DOWN_COMPACT=1; both down-prefix paths read the compact copy)")
