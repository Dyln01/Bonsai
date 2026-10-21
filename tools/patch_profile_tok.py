#!/usr/bin/env python3
"""patch_profile_tok.py -- Phase 4 data pipeline: per-token (x, top-K) dumps.

Design contract: docs/PREDICTOR_QWEN36.md. The Deja Vu-class predictor needs
per-TOKEN training pairs, which HOT_PROFILE does not keep (it accumulates
corpus sums). This adds a one-off profiling mode:

  Q36_PROFILE_TOK=<path prefix>  -> one raw binary per layer,
      <prefix>_L<layer>.bin, record = x[D] f32 | K u32 indices | K f32 energies
  Q36_PROFILE_TOK_K=<n>          -> K (default 64)

The hook sits beside the two existing q36_hot_acc_add call sites (the S=1
decode path and the batched prefill path of the dense MLP), so a single
prompt captures prefill and decode rows alike. The energies are the
post-silu silu(g)*u values the MLP just computed; x is the same row the
MLP consumed (post post_ln). Runs are meant to be FULL compute: HOT_FRAC
unset (equals 1.0) and GRIFFIN=0 -- anything less would dump a truncated
activation vector and bias the predictor's targets.

The selector is a fixed-K insertion scan (most values fail one compare),
so the dump costs about O(I + K*K) per row, not a sort. stdio buffers
are flushed at exit(0); a killed run loses the tail (acceptable for a
one-off; the calibration corpus is cheap to re-run). Debug messages
carry no trailing newline on purpose: this patcher is generated through
tool transports that mangle backslash-n inside C strings, so the escape
simply is not used anywhere in the emitted code.

Knob-off = zero code-path change (one pointer check per MLP block).
Backup qwen36.c.preptk. Refuses to run twice.
Usage: python3 patch_profile_tok.py <path-to-qwen36.c>
"""
import os, shutil, sys

EDITS = []
def edit(old, new): EDITS.append((old, new))

# ------------------------------------------------------------- E1: dump machinery
edit(
'static void q36_hot_acc_add(int layer, const float *g, const float *u, int n) {\n',
'/* ---- Phase 4: per-token profiling dump -- see docs/PREDICTOR_QWEN36.md.\n'
' * Sibling of the corpus-sum accumulator above: keeps what a Deja Vu-class\n'
' * predictor needs -- the layer INPUT row and the true per-token top-K\n'
' * neurons with their energies, from the FULL compute the MLP just did. */\n'
'static char   g_tok_dump_path[1024] = {0};\n'
'static int    g_tok_dump_k = 64;\n'
'#define Q36_TOK_MAXI 16384\n'
'static void q36_tok_dump(int layer, const float *x, const float *en, int D, int I) {\n'
'    if (!g_tok_dump_path[0]) return;\n'
'    static FILE *tf[512];\n'
'    static float ce[Q36_TOK_MAXI]; static uint32_t ci[Q36_TOK_MAXI];\n'
'    if (layer < 0 || layer >= 512 || I <= 0 || I > Q36_TOK_MAXI) return;\n'
'    int K = g_tok_dump_k < I ? g_tok_dump_k : I;\n'
'    /* fixed-K insertion scan: most rows fail the first compare */\n'
'    int n = 0;\n'
'    for (int i = 0; i < I; i++) {\n'
'        float e = en[i];\n'
'        if (n < K || e > ce[K-1]) {\n'
'            int j = n < K ? n++ : K-1;\n'
'            while (j > 0 && ce[j-1] < e) { ce[j] = ce[j-1]; ci[j] = ci[j-1]; j--; }\n'
'            ce[j] = e; ci[j] = (uint32_t)i;\n'
'        }\n'
'    }\n'
'    if (!tf[layer]) {\n'
'        char pth[1200]; snprintf(pth, sizeof pth, "%s_L%d.bin", g_tok_dump_path, layer);\n'
'        tf[layer] = fopen(pth, "ab");\n'
'        if (!tf[layer]) { fprintf(stderr, "[tok-dump] cannot open: %s", pth); g_tok_dump_path[0] = 0; return; }\n'
'    }\n'
'    fwrite(x, sizeof(float), (size_t)D, tf[layer]);\n'
'    fwrite(ci, sizeof(uint32_t), (size_t)K, tf[layer]);\n'
'    fwrite(ce, sizeof(float), (size_t)K, tf[layer]);\n'
'}\n'
'static void q36_hot_acc_add(int layer, const float *g, const float *u, int n) {\n'
)

# ------------------------------------------------------------- E2: S=1 path hook
edit(
'            { int _li = (int)(l - m->L); q36_hot_acc_add(_li, g, u, I); }\n',
'            { int _li = (int)(l - m->L); q36_hot_acc_add(_li, g, u, I); }\n'
'            { int _li2 = (int)(l - m->L); q36_tok_dump(_li2, xs, g, D, I); }\n'
)

# ------------------------------------------------------------- E3: batched path hook
edit(
'            { int _li = (int)(l - m->L); q36_hot_acc_add(_li, bg, bu, I); }\n',
'            { int _li = (int)(l - m->L); q36_hot_acc_add(_li, bg, bu, I); }\n'
'            for (int s = 0; s < rows; s++)\n'
'                q36_tok_dump((int)(l - m->L), x + (int64_t)(base+s)*D, bg + (int64_t)s*I, D, I);\n'
)

# ------------------------------------------------------------- E4: knob parse
edit(
'    { const char *hp = getenv("HOT_PROFILE"); if (hp && *hp) snprintf(g_hot_profile_path, sizeof g_hot_profile_path, "%s", hp); }\n',
'    { const char *hp = getenv("HOT_PROFILE"); if (hp && *hp) snprintf(g_hot_profile_path, sizeof g_hot_profile_path, "%s", hp); }\n'
'    { const char *tp = getenv("Q36_PROFILE_TOK");\n'
'      if (tp && *tp) { snprintf(g_tok_dump_path, sizeof g_tok_dump_path, "%s", tp);\n'
'        const char *kk = getenv("Q36_PROFILE_TOK_K");\n'
'        if (kk && *kk) { g_tok_dump_k = atoi(kk); if (g_tok_dump_k < 1) g_tok_dump_k = 64; }\n'
'        fprintf(stderr, "[tok-dump] per-token pairs (K=%d) -> %s", g_tok_dump_k, g_tok_dump_path); } }\n'
)

def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
    src = open(path, "r", encoding="utf-8", newline="").read()
    if "q36_tok_dump" in src:
        sys.exit("refusing: qwen36.c already carries q36_tok_dump (idempotence guard)")
    for i, (old, new) in enumerate(EDITS):
        n = src.count(old)
        if n != 1:
            sys.exit("E%02d: anchor count=%d (need 1): %.90r" % (i + 1, n, old))
        src = src.replace(old, new, 1)
    bak = path + ".preptk"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print("backup -> %s" % bak)
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    print("applied %d edits -> %s" % (len(EDITS), path))

if __name__ == "__main__":
    main()
