#!/usr/bin/env python3
"""patch_kvq_glm53.py — Phase 4c slice 4b: KV_TQ on glm53 MLA latent rows.

The MLA cache holds one latent row per position (kv_lora floats); kv_b is
absorbed at load and queries enter latent space per head, so the 3b-rot
pattern is absorption-native here: rotate the absorbed per-head query once,
dot against rotated-space levels, scale by the hoisted radius, accumulate the
weighted latent mean in rotated space and unrotate once per (token, head).
Codec 0 (polar) falls back to full per-row dequant (still once per position,
shared by all heads).

Knobs: KV_TQ=<bits 2..8>, KV_TQ_CODEC=0|1, GLM53_TQ_ABSORB=0|1 (default 1).
Slice-4b limits (enforced): segment snapshots refuse KV_TQ (the span exposes
latent as floats); serve/mux untouched (glm53 prompt/TF modes are the gates).

Gates (CI): tq-off patched == stock (DUMP bytes); tq-on self-deterministic;
absorb 0-vs-1 cosine >= 0.9999; tq-on vs off prefill cosine in the codec band.

Usage: python3 patch_kvq_glm53.py <colibri-c-dir>
Backup: glm53.c.q53pretq. Refuses twice.
"""
import os
import shutil
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/colibri/c")
P = os.path.join(D, "glm53.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "g_g53_tq_bits" in src:
    raise SystemExit("patch_kvq_glm53: already applied")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# globals + include, before the MLA section
edit("""/* ---------- MLA + indexer con k-pool ---------- */
""",
     """/* ---------- Bonsai KV_TQ on MLA latents (patch_kvq_glm53) ---------- */
#include "kv_tq.h"
static int g_g53_tq_bits = 0, g_g53_tq_codec = 1, g_g53_tq_absorb = 1;
static inline void g53_tq_rot_noscale(const uint8_t *s, float *dst, int n) {
    float inv = 1.f / sqrtf((float)n);
    for (int i = 0; i < n; i++) { int c = (s[i >> 1] >> ((i & 1) * 4)) & 0xF; dst[i] = coli_q4_lev[c] * inv; }
}

/* ---------- MLA + indexer con k-pool ---------- */
""")

# state fields
edit("""    float *latent;                        /* [cap][kv_lora]: MLA assorbita */
""",
     """    float *latent;                        /* [cap][kv_lora]: MLA assorbita */
    uint8_t *latq; float *tscale;         /* Bonsai KV_TQ: packed latents + radii */
""")

# alloc
edit("""            st->latent = malloc((size_t)cap * c->kv_lora * sizeof(float));
            st->ikeys = malloc((size_t)cap * c->index_hd * sizeof(float));
            st->igates = malloc((size_t)cap * c->index_hd * sizeof(float));
            if (!st->latent || !st->ikeys || !st->igates) {
""",
     """            if (g_g53_tq_bits) {
                st->latq = malloc((size_t)cap * coli_kvq_row_bytes(c->kv_lora, g_g53_tq_bits, g_g53_tq_codec));
                st->tscale = malloc((size_t)cap * sizeof(float));
            } else {
                st->latent = malloc((size_t)cap * c->kv_lora * sizeof(float));
            }
            st->ikeys = malloc((size_t)cap * c->index_hd * sizeof(float));
            st->igates = malloc((size_t)cap * c->index_hd * sizeof(float));
            if ((g_g53_tq_bits ? (!st->latq || !st->tscale) : !st->latent) || !st->ikeys || !st->igates) {
""")

# locals in mla_layer
edit("""    float *latent = st->latent;           /* tutto il prefisso, non solo i nuovi */
""",
     """    float *latent = st->latent;           /* tutto il prefisso, non solo i nuovi */
    uint8_t *latq = st->latq; float *tscale = st->tscale;
    int tq = g_g53_tq_bits && latq;
    int absorb = tq && g_g53_tq_codec == 1 && g_g53_tq_absorb;
    int rowb = tq ? coli_kvq_row_bytes(L, g_g53_tq_bits, g_g53_tq_codec) : 0;
""")

# write side
edit("""    mm(latent + (size_t)base * L, &l->kva, x, tokens);
    for (int t = 0; t < tokens; t++) {
        float *here = latent + (size_t)(base + t) * L;
        rms(here, here, l->kva_ln, L, c->eps);
    }
""",
     """    if (tq) {
        float *tmp = malloc((size_t)tokens * L * sizeof(float));
        if (!tmp) { fprintf(stderr, "OOM in MLA tq\\n"); exit(1); }
        mm(tmp, &l->kva, x, tokens);
        for (int t = 0; t < tokens; t++) {
            float *here = tmp + (size_t)t * L;
            rms(here, here, l->kva_ln, L, c->eps);
            tscale[base + t] = coli_kvq_quant_row(here, latq + (size_t)(base + t) * rowb,
                                                  L, g_g53_tq_bits, g_g53_tq_codec);
        }
        free(tmp);
    } else {
        mm(latent + (size_t)base * L, &l->kva, x, tokens);
        for (int t = 0; t < tokens; t++) {
            float *here = latent + (size_t)(base + t) * L;
            rms(here, here, l->kva_ln, L, c->eps);
        }
    }
""")

# read side: per-position dequant/levels shared by all heads
edit("""    for (int t = 0; t < tokens; t++) {
        const int *chosen = selected + (size_t)t * width;
        for (int h = 0; h < H; h++) {
            const float *q = absorbed + ((size_t)t * H + h) * L;
""",
     """    for (int t = 0; t < tokens; t++) {
        const int *chosen = selected + (size_t)t * width;
        float *lev = tq ? malloc((size_t)width * L * sizeof(float)) : NULL;
        float *rad = tq ? malloc((size_t)width * sizeof(float)) : NULL;
        if (tq && (!lev || !rad)) { fprintf(stderr, "OOM in MLA tq read\\n"); exit(1); }
        if (tq) for (int i = 0; i < width; i++) {
            const int at = chosen[i];
            if (at < 0 || at >= seen) { rad[i] = 0.f; continue; }
            if (absorb) { g53_tq_rot_noscale(latq + (size_t)at * rowb, lev + (size_t)i * L, L); rad[i] = tscale[at]; }
            else { coli_kvq_dequant_row(latq + (size_t)at * rowb, tscale[at], lev + (size_t)i * L, L, g_g53_tq_bits, g_g53_tq_codec); rad[i] = 1.f; }
        }
        for (int h = 0; h < H; h++) {
            const float *q = absorbed + ((size_t)t * H + h) * L;
            float *qrh = NULL;
            if (absorb) { qrh = malloc((size_t)L * sizeof(float)); if (!qrh) { fprintf(stderr, "OOM qr\\n"); exit(1); } coli_tq_rotate(q, qrh, L, COLI_TQ_SEED); }
""")

# score dot
edit("""                const float *c_j = latent + (size_t)at * L;
                float dot = 0.0f;
                for (int d = 0; d < L; d++) dot += q[d] * c_j[d];
                score[used] = dot * scale;
""",
     """                const float *c_j = tq ? lev + (size_t)i * L : latent + (size_t)at * L;
                float dot = 0.0f;
                const float *qq = (tq && absorb) ? qrh : q;
                for (int d = 0; d < L; d++) dot += qq[d] * c_j[d];
                score[used] = dot * scale * ((tq && absorb) ? rad[i] : 1.f);
""")

# pooled accumulation in rotated space + unrotate
edit("""                const float weight = (float)(score[seen_slot++] / total);
                const float *c_j = latent + (size_t)at * L;
                for (int d = 0; d < L; d++) pool[d] += weight * c_j[d];
            }
        }
    }
""",
     """                const float weight = (float)(score[seen_slot++] / total);
                const float *c_j = tq ? lev + (size_t)i * L : latent + (size_t)at * L;
                float wj = weight * ((tq && absorb) ? rad[i] : 1.f);
                for (int d = 0; d < L; d++) pool[d] += wj * c_j[d];
            }
            if (absorb) coli_tq_unrotate(pool, L, COLI_TQ_SEED);
            free(qrh);
        }
        free(lev); free(rad);
    }
""")

# free
edit("""    free(st->latent); free(st->ikeys); free(st->igates);
""",
     """    free(st->latent); free(st->latq); free(st->tscale); free(st->ikeys); free(st->igates);
""")

# segment span refusal
edit("""            spans[count++] = (ColiSegmentStateSpan){ st->latent, cap * (size_t)c->kv_lora * sizeof(float) };
""",
     """            if (st->latq) return coli_segment_adapter_error(error, error_size,
                "KV_TQ latents are not segment-restorable in slice 4b");
            spans[count++] = (ColiSegmentStateSpan){ st->latent, cap * (size_t)c->kv_lora * sizeof(float) };
""")

# parse in main
edit("""    coli_omp_tune_threads("glm53");
""",
     """    coli_omp_tune_threads("glm53");
    { const char *te = getenv("KV_TQ"); int bits = te && *te ? atoi(te) : 0;
      if (bits && (bits < 2 || bits > 8)) { fprintf(stderr, "[kvq] KV_TQ must be 2..8\\n"); return 2; }
      g_g53_tq_bits = bits;
      const char *ce = getenv("KV_TQ_CODEC"); g_g53_tq_codec = (ce && *ce == '0') ? 0 : 1;
      const char *ae = getenv("GLM53_TQ_ABSORB"); if (ae && ae[0] == '0') g_g53_tq_absorb = 0; }
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_kvq_glm53: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".q53pretq"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_kvq_glm53 applied (%d edits, backup %s)" % (len(EDITS), bak))
