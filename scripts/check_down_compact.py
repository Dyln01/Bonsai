#!/usr/bin/env python3
"""check_down_compact.py — gate for engine patch E28 (Q36_DOWN_COMPACT).

Extracts the down-prefix functions VERBATIM from the two engine snapshots (qwen36.c.predcmp = before E28/E29:
scalar staging, strided reads; qwen36.c = after: compact buffer + AVX2 staging), compiles both into one harness against a REAL mmap'd file (so madvise acts on a genuine
mapping), and requires bit-identical outputs for single-token and batched calls, odd and even Ik, with
compaction OFF (the default) and ON (Q36_DOWN_COMPACT=1). The harness clears the compact cache when it
unmaps a test matrix, mirroring the engine invariant that mappings are never unmapped (so an address can
never be reused for different bytes); without that, address reuse across test matrices would hand back a
stale copy.   usage: python3 scripts/check_down_compact.py [--mutate NAME]
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "..", "engine")

def grab(path, marker):
    s = open(path, "rb").read().decode("utf8", "surrogateescape")
    a = s.index(marker); b = s.index("/* ---- Phase 4b: resident neuron-row window", a)
    return s[a:b]

HARNESS_HEAD = r'''
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
typedef struct { const float *w; int8_t *q; float *sc; int I, O; uint8_t *q4; float *sg; int ng;
                 void *map; int64_t map_len; void *vk; int vk_off; uint16_t *h;
                 char *vk_name; const char *vk_tag; int vk_quant, vk_fmt, vk_gone, vk_imported; } QW;
static float map_f16_one(uint16_t h) { return (float)h; }
#include "mapi4_avx2.h"   /* q36_m4_stage_prefix (E29) — the engine includes this header ahead of these functions */
static float dot_f32_lanes(const float *a, const float *b, int n) {
    float l[8] = {0,0,0,0,0,0,0,0}; int i = 0;
    for (; i + 8 <= n; i += 8) for (int k = 0; k < 8; k++) l[k] += a[i+k]*b[i+k];
    float s = ((l[0]+l[4])+(l[1]+l[5]))+((l[2]+l[6])+(l[3]+l[7]));
    for (; i < n; i++) s += a[i]*b[i]; return s;
}
'''
HARNESS_MAIN = r'''
static uint32_t r_ = 777; static uint32_t rnd(void) { r_ ^= r_ << 13; r_ ^= r_ >> 17; r_ ^= r_ << 5; return r_; }
int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/tmp";
    const int Ds[] = {1, 7, 64, 130}, Is[] = {64, 130, 1024};
    int cases = 0, bad = 0;
    for (int di = 0; di < 4; di++) for (int ii = 0; ii < 3; ii++) {
        int D = Ds[di], I = Is[ii]; size_t bytes = (size_t)D * (I / 2);
        char path[512]; snprintf(path, sizeof path, "%s/mat_%d_%d.bin", dir, D, I);
        int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        uint8_t *tmp = malloc(bytes); for (size_t k = 0; k < bytes; k++) tmp[k] = rnd();
        if (write(fd, tmp, bytes) != (ssize_t)bytes) return 9;
        free(tmp);
        void *map = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0); if (map == MAP_FAILED) return 8;
        float *sc = malloc(D * 4); for (int d = 0; d < D; d++) sc[d] = (d % 3 == 0) ? 1.0f : 0.002f + 0.0003f * d;
        QW dw; memset(&dw, 0, sizeof dw); dw.map = map; dw.map_len = (int64_t)D * I / 2; dw.sc = sc; dw.I = I; dw.O = D;
        const int Iks[] = {1, 2, 3, 33, 64, I / 2, I - 1, I};
        for (int ki = 0; ki < 8; ki++) { int Ik = Iks[ki]; if (Ik < 1 || Ik > I) continue;
            for (int S = 1; S <= 4; S++) {
                float *g = malloc((size_t)S * Ik * 4); for (int i = 0; i < S * Ik; i++) g[i] = ((int)(rnd() % 2001) - 1000) / 400.0f;
                float *ya = calloc((size_t)S * D, 4), *yb = calloc((size_t)S * D, 4);
                old_down_batch(&dw, ya, g, S, D, I, Ik); new_down_batch(&dw, yb, g, S, D, I, Ik);
                cases++; if (memcmp(ya, yb, (size_t)S * D * 4)) { bad++; if (bad < 4) printf("BATCH DIFF D=%d I=%d Ik=%d S=%d\n", D, I, Ik, S); }
                if (S == 1) { memset(ya, 0, D * 4); memset(yb, 0, D * 4);
                    old_down_single(&dw, ya, g, D, I, Ik); new_down_single(&dw, yb, g, D, I, Ik);
                    cases++; if (memcmp(ya, yb, (size_t)D * 4)) { bad++; if (bad < 4) printf("SINGLE DIFF D=%d I=%d Ik=%d\n", D, I, Ik); } }
                free(g); free(ya); free(yb);
            }
        }
        munmap(map, bytes); close(fd); free(sc); unlink(path);
        for (int k = 0; k < g_q36_dc_n; k++) free(g_q36_dc[k].buf);   /* mapping gone: drop its compact copies */
        g_q36_dc_n = 0;
    }
    printf("%d down-prefix calls (batch S=1..4 + single, odd/even Ik, real mmap): %d differ\n", cases, bad);
    return bad ? 1 : 0;
}
'''

def build(workdir, mutate=None):
    old = grab(os.path.join(ENG, "qwen36.c.predcmp"), "static void q36_down_prefix_batch(")
    new = grab(os.path.join(ENG, "qwen36.c"), "/* ---- Q36_DOWN_COMPACT=1")
    if mutate == "floor_stride":     # compact rows one byte short for odd Ik
        assert "int64_t cb = (Ik + 1) >> 1;" in new; new = new.replace("int64_t cb = (Ik + 1) >> 1;", "int64_t cb = Ik >> 1;", 1)
    if mutate == "wrong_stride":     # reads the compact buffer with the original stride
        assert "*stride = cb;\n    return buf;" in new; new = new.replace("*stride = cb;\n    return buf;", "*stride = I >> 1;\n    return buf;", 1)
    if mutate == "skip_copy":        # compaction that forgets to copy
        assert "memcpy(buf + (int64_t)d * cb" in new; new = new.replace("memcpy(buf + (int64_t)d * cb", "memset(buf + (int64_t)d * cb, 0", 1)
        new = new.replace("mp + (int64_t)d * (I >> 1), (size_t)cb);", "(size_t)cb);", 1)
    if mutate in ("prefix_odd_noscale", "prefix_wrong_nibble"):   # mutations of the E29 header helper
        h = open(os.path.join(ENG, "mapi4_avx2.h")).read()
        if mutate == "prefix_odd_noscale":
            a = "float v = (float)e; if (rs != 1.0f) v *= rs;"; assert a in h; h = h.replace(a, "float v = (float)e;", 1)
        else:
            a = "int8_t e = (int8_t)(b & 0xF); if (e & 8) e -= 16;\n        float v"; assert a in h
            h = h.replace(a, "int8_t e = (int8_t)(b >> 4); if (e & 8) e -= 16;\n        float v", 1)
        os.makedirs(os.path.join(workdir, "inc"), exist_ok=True)
        open(os.path.join(workdir, "inc", "mapi4_avx2.h"), "w").write(h)
    old = old.replace("q36_down_prefix_batch", "old_down_batch").replace("q36_down_prefix(", "old_down_single(")
    new = new.replace("q36_down_prefix_batch", "new_down_batch").replace("q36_down_prefix(", "new_down_single(")
    c = os.path.join(workdir, "h.c")
    open(c, "w").write(HARNESS_HEAD + old + "\n" + new + HARNESS_MAIN)
    exe = os.path.join(workdir, "h")
    inc = os.path.join(workdir, "inc") if os.path.isdir(os.path.join(workdir, "inc")) else ENG
    r = subprocess.run(["gcc", "-O2", "-fopenmp", "-I", inc, "-o", exe, c, "-lm"], capture_output=True, text=True)
    if r.returncode:
        print("FAIL  harness did not compile:\n" + r.stderr[-1500:]); sys.exit(2)
    return exe

def main():
    mutate = sys.argv[sys.argv.index("--mutate") + 1] if "--mutate" in sys.argv else None
    rc = 0
    with tempfile.TemporaryDirectory() as w:
        exe = build(w, mutate)
        for label, env in (("compaction OFF (default)", {}), ("compaction ON (Q36_DOWN_COMPACT=1)", {"Q36_DOWN_COMPACT": "1"})):
            r = subprocess.run([exe, w], capture_output=True, text=True, env=dict(os.environ, **env))
            out = (r.stdout.strip().splitlines() or ["(no output)"])
            print("%-4s %-36s %s" % ("ok" if r.returncode == 0 else "FAIL", label, out[-1]))
            if r.returncode: rc = 1; [print("       " + l) for l in out[:-1]]
    print("down-compact gate: " + ("PASS" if rc == 0 else "FAIL"))
    sys.exit(rc)

main()
