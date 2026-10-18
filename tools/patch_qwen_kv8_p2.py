#!/usr/bin/env python3
"""Apply part 2 of the qwen36 KV8 patch (idempotent-ish, all assertions local)."""
import os
import sys
import re

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P).read()
changed = []

def rep(old, new, tag):
    global src
    assert old in src, f"MISSING: {tag}"
    src = src.replace(old, new, 1)
    changed.append(tag)

# A. Q36Seq struct
rep("""typedef struct {
    float **K, **V, **DN_rec, **DN_conv;
    int kv_len;""",
    """typedef struct {
    float **K, **V, **DN_rec, **DN_conv;
    float **Kscale, **Vscale;   /* KV8: per-row f32 scales, parallel to K/V */
    int kv_len;""", "A: Q36Seq")

# B. swap
rep("    Q36_SEQ_SWAP(float **, m->K, q->K); Q36_SEQ_SWAP(float **, m->V, q->V);",
    "    Q36_SEQ_SWAP(float **, m->K, q->K); Q36_SEQ_SWAP(float **, m->V, q->V);\n"
    "    Q36_SEQ_SWAP(float **, m->Kscale, q->Kscale); Q36_SEQ_SWAP(float **, m->Vscale, q->Vscale);",
    "B: swap")

# C. free
rep("""        if (q->K) free(q->K[i]); if (q->V) free(q->V[i]);
        if (q->DN_rec) free(q->DN_rec[i]); if (q->DN_conv) free(q->DN_conv[i]);
    }
    free(q->K); free(q->V); free(q->DN_rec); free(q->DN_conv);""",
    """        if (q->K) free(q->K[i]); if (q->V) free(q->V[i]);
        if (q->Kscale) free(q->Kscale[i]); if (q->Vscale) free(q->Vscale[i]);
        if (q->DN_rec) free(q->DN_rec[i]); if (q->DN_conv) free(q->DN_conv[i]);
    }
    free(q->K); free(q->V); free(q->Kscale); free(q->Vscale); free(q->DN_rec); free(q->DN_conv);""",
    "C: free")

# D. alloc
rep("""    q->K = calloc((size_t)n, sizeof(float *)); q->V = calloc((size_t)n, sizeof(float *));
    q->DN_rec = calloc((size_t)n, sizeof(float *)); q->DN_conv = calloc((size_t)n, sizeof(float *));
    int ok = q->K && q->V && q->DN_rec && q->DN_conv && kv_prefix_alloc(&q->kvp, m->kv_cap);
    for (int i = 0; ok && i < n; i++) {
        if (c->is_attn[i]) {
            size_t kv = (size_t)c->kv_heads * m->kv_cap * c->k_head_dim;
            q->K[i] = malloc(kv * sizeof(float)); q->V[i] = malloc(kv * sizeof(float));
            ok = q->K[i] && q->V[i];
        } else if (m->DN_rec[i]) {""",
    """    q->K = calloc((size_t)n, sizeof(float *)); q->V = calloc((size_t)n, sizeof(float *));
    if (m->kv8) { q->Kscale = calloc((size_t)n, sizeof(float *)); q->Vscale = calloc((size_t)n, sizeof(float *)); }
    q->DN_rec = calloc((size_t)n, sizeof(float *)); q->DN_conv = calloc((size_t)n, sizeof(float *));
    int ok = q->K && q->V && (!m->kv8 || (q->Kscale && q->Vscale)) && q->DN_rec && q->DN_conv && kv_prefix_alloc(&q->kvp, m->kv_cap);
    for (int i = 0; ok && i < n; i++) {
        if (c->is_attn[i]) {
            size_t kv = (size_t)c->kv_heads * m->kv_cap * c->k_head_dim;
            if (m->kv8) {
                q->K[i] = malloc(kv); q->V[i] = malloc(kv);   /* 1 B/element, fp8 */
                q->Kscale[i] = malloc((size_t)c->kv_heads * m->kv_cap * sizeof(float));
                q->Vscale[i] = malloc((size_t)c->kv_heads * m->kv_cap * sizeof(float));
                ok = q->K[i] && q->V[i] && q->Kscale[i] && q->Vscale[i];
            } else {
                q->K[i] = malloc(kv * sizeof(float)); q->V[i] = malloc(kv * sizeof(float));
                ok = q->K[i] && q->V[i];
            }
        } else if (m->DN_rec[i]) {""", "D: alloc")

# E1. ensure_kv: keep old scale pointers
rep("    float **oldK = m->K, **oldV = m->V;",
    "    float **oldK = m->K, **oldV = m->V;\n"
    "    float **oldKscale = m->Kscale, **oldVscale = m->Vscale;\n"
    "    m->Kscale = NULL; m->Vscale = NULL;", "E1: oldKscale")

# E2. ensure_kv: free old scales
rep("""    if (oldK){
        for (int i = 0; i < c->n_layers; i++){ if (oldK[i]) free(oldK[i]); if (oldV[i]) free(oldV[i]); }
        free(oldK); free(oldV);
    }""",
    """    if (oldK){
        for (int i = 0; i < c->n_layers; i++){ if (oldK[i]) free(oldK[i]); if (oldV[i]) free(oldV[i]); }
        free(oldK); free(oldV);
    }
    if (oldKscale){
        for (int i = 0; i < c->n_layers; i++){ if (oldKscale[i]) free(oldKscale[i]); if (oldVscale[i]) free(oldVscale[i]); }
        free(oldKscale); free(oldVscale);
    }""", "E2: free scales")

# F. fix lut init name
if "coli_kv8_lut_init()" in src:
    src = src.replace("coli_kv8_lut_init()", "coli_fp8_lut_init()")
    changed.append("F: lut name")

# G. fix possibly-broken fprintf literal: find the broken (raw newline) version
bad = re.compile(r'fprintf\(stderr, "\[qwen36\] KV8=1: KV rows stored fp8-e4m3 \+ per-row f32 scale \(lossy, opt-in\)\n"\);', re.S)
if bad.search(src):
    src = bad.sub('fprintf(stderr, "[qwen36] KV8=1: KV rows stored fp8-e4m3 + per-row f32 scale (lossy, opt-in)\\n");', src)
    changed.append("G: fixed string literal")

open(P, "w").write(src)
print("applied:", ", ".join(changed))
