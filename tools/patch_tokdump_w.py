#!/usr/bin/env python3
"""patch_tokdump_w.py — per-token dumps from the narrowed PAGEBATCH path.

Q36_PROFILE_TOK's writer only fired in the per-row (B==1) fallback; once the
decode gather went live (Phase 4b slice 1) and prefill runs PAGEBATCH, dumps
went silent — which would have silently broken the 4a re-run and the window
sim. Adds q36_tok_dump_w(layer, x_row, en_narrowed, width, I, D): top-K over
the hot-prefix energies (prefix indices ARE original indices), same record
format (x[D] f32, ci[K] u32, ce[K] f32) so exp_window_sim.py and
exp_predictor_q36.py read both paths' dumps unchanged.

Anchored exact-byte edit, house style. Backup qwen36.c.pretdw.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "q36_tok_dump_w" in src:
    raise SystemExit("patch_tokdump_w: already applied")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

edit("""fwrite(ce, sizeof(float), (size_t)K, tf[layer]);
}
""",
     """fwrite(ce, sizeof(float), (size_t)K, tf[layer]);
}
/* Phase 4b slice 2: same record format, energies over the hot prefix only
 * (width < I); prefix indices are original indices, so downstream sim/predictor
 * scripts need no changes. Separate FILE* table: both writers append. */
static void q36_tok_dump_w(int layer, const float *x, const float *en, int width, int I, int D) {
    static FILE *tfw[512];
    static float cew[Q36_TOK_MAXI]; static uint32_t ciw[Q36_TOK_MAXI];
    if (!g_tok_dump_path[0]) return;
    if (layer < 0 || layer >= 512 || I <= 0 || width <= 0 || width > Q36_TOK_MAXI) return;
    int K = g_tok_dump_k < width ? g_tok_dump_k : width;
    int n = 0;
    for (int i = 0; i < width; i++) {
        float e = en[i];
        if (n < K || e > cew[K-1]) {
            int j = n < K ? n++ : K-1;
            while (j > 0 && cew[j-1] < e) { cew[j] = cew[j-1]; ciw[j] = ciw[j-1]; j--; }
            cew[j] = e; ciw[j] = (uint32_t)i;
        }
    }
    if (!tfw[layer]) {
        char pth[1200]; snprintf(pth, sizeof pth, "%s_L%d.bin", g_tok_dump_path, layer);
        tfw[layer] = fopen(pth, "ab");
        if (!tfw[layer]) { fprintf(stderr, "[tok-dump] cannot open: %s", pth); g_tok_dump_path[0] = 0; return; }
    }
    fwrite(x, sizeof(float), (size_t)D, tfw[layer]);
    fwrite(ciw, sizeof(uint32_t), (size_t)K, tfw[layer]);
    fwrite(cew, sizeof(float), (size_t)K, tfw[layer]);
}
""")

edit("""                if (S > 1) q36_griff_observe_w((int)(l - m->L), bg, rows, Ik, Ik, I);   /* Phase 4b: dense GRIFFIN */
""",
     """                if (S > 1) q36_griff_observe_w((int)(l - m->L), bg, rows, Ik, Ik, I);   /* Phase 4b: dense GRIFFIN */
                if (S > 1) for (int s2 = 0; s2 < rows; s2++)
                    q36_tok_dump_w((int)(l - m->L), x + (int64_t)(base + s2) * D, bg + (int64_t)s2 * Ik, Ik, I, D);
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_tokdump_w: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".pretdw"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_tokdump_w applied (%d edits, backup %s)" % (len(EDITS), bak))
