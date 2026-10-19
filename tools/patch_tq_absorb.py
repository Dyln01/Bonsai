#!/usr/bin/env python3
"""patch_tq_absorb.py — Phase 3b-rot: absorbed-rotation decode for KV_TQ codec 1.

Design contract: docs/ABSORB_ROT_QWEN36.md. The fixed orthogonal rotation
(sign flips + self-inverse FWHT, COLI_TQ_SEED) moves OUT of the per-row decode
loop: queries are rotated once per (head, step), K rows dequantize to raw
rotated-space levels (no FWHT), scores dot against rotated queries, V rows
accumulate in rotated space and are unrotated once per (head, step).
Per-row cost becomes KV8-class; math identical up to fp re-association.

Knob Q36_TQ_ABSORB=0 restores the 3a loop verbatim. Active only when
kv_tq && tq_codec==1 && q_per_kv*kvd <= 8192.

Anchored exact-byte edits, house style. Backup qwen36.c.preabsorb.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "g_q36_tq_absorb" in src:
    raise SystemExit("patch_tq_absorb: already applied")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

edit("""static int  g_q36_progress = 1;   /* Q36_PROGRESS=0 disables prefill progress lines */
""",
     """static int  g_q36_progress = 1;   /* Q36_PROGRESS=0 disables prefill progress lines */
static int  g_q36_tq_absorb = 1;  /* Q36_TQ_ABSORB=0: 3a loop verbatim (A/B + fallback) */
""")

edit("""    { const char *pv = getenv("Q36_PROGRESS"); if (pv && pv[0] == '0') g_q36_progress = 0; }
""",
     """    { const char *pv = getenv("Q36_PROGRESS"); if (pv && pv[0] == '0') g_q36_progress = 0; }
    { const char *av = getenv("Q36_TQ_ABSORB"); if (av && av[0] == '0') g_q36_tq_absorb = 0; }
""")

edit("""static inline void q36_kv_dequant_noscale(const uint8_t *src, float *dst, int kvd, int tq, int bits, int codec) {
    if (tq) coli_kvq_dequant_row(src, 1.f, dst, kvd, bits, codec);
    else kv8_dequant_noscale(src, dst, kvd);
}
""",
     """static inline void q36_kv_dequant_noscale(const uint8_t *src, float *dst, int kvd, int tq, int bits, int codec) {
    if (tq) coli_kvq_dequant_row(src, 1.f, dst, kvd, bits, codec);
    else kv8_dequant_noscale(src, dst, kvd);
}
/* Phase 3b-rot: codec-1 row in ROTATED space, radius 1 (levels/√n), no FWHT —
 * the absorbed-rotation decode dots these against rotated queries and
 * unrotates the accumulated output once per head per step. */
static inline void q36_q4_rot_noscale(const uint8_t *src, float *dst, int n) {
    float inv = 1.f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        int c = (src[i >> 1] >> ((i & 1) * 4)) & 0xF;
        dst[i] = coli_q4_lev[c] * inv;
    }
}
""")

edit("""            float *cxg = vq + (int64_t)KV8_TILE_ROWS * kvd;   /* q_per_kv*kvd tail of the grown staging */
""",
     """            float *cxg = vq + (int64_t)KV8_TILE_ROWS * kvd;   /* q_per_kv*kvd tail of the grown staging */
            int absorb = m->kv_tq && m->tq_codec == 1 && g_q36_tq_absorb
                         && (q_per_kv * kvd) <= 8192;          /* Phase 3b-rot */
            float qrg[8192];
""")

edit("""            for (int g = 0; g < q_per_kv; g++)
                for (int dd = 0; dd < kvd; dd++) cxg[(int64_t)g*kvd + dd] = 0.f;
""",
     """            for (int g = 0; g < q_per_kv; g++)
                for (int dd = 0; dd < kvd; dd++) cxg[(int64_t)g*kvd + dd] = 0.f;
            if (absorb)
                for (int g = 0; g < q_per_kv; g++)
                    coli_tq_rotate(query + ((int64_t)s*H + (kvh*q_per_kv + g))*hd,
                                   qrg + (size_t)g*kvd, kvd, COLI_TQ_SEED);
""")

edit("""                for (int t = 0; t < tn; t++)
                    q36_kv_dequant_noscale(K8 + (base + t0 + t)*rb,
                                           kq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
""",
     """                for (int t = 0; t < tn; t++) {
                    if (absorb) q36_q4_rot_noscale(K8 + (base + t0 + t)*rb, kq + (int64_t)t*kvd, kvd);
                    else q36_kv_dequant_noscale(K8 + (base + t0 + t)*rb,
                                                kq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
                }
""")

edit("""                        sc[t0 + t] = dot_f32_lanes(qv, kq + (int64_t)t*kvd, kvd)
                                     * (Ksc[base + t0 + t] * scale);
""",
     """                        sc[t0 + t] = dot_f32_lanes(absorb ? qrg + (size_t)g*kvd : qv,
                                                    kq + (int64_t)t*kvd, kvd)
                                     * (Ksc[base + t0 + t] * scale);
""")

edit("""                for (int t = 0; t < tn; t++)
                    q36_kv_dequant_noscale(V8 + (base + t0 + t)*rb,
                                           vq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
""",
     """                for (int t = 0; t < tn; t++) {
                    if (absorb) q36_q4_rot_noscale(V8 + (base + t0 + t)*rb, vq + (int64_t)t*kvd, kvd);
                    else q36_kv_dequant_noscale(V8 + (base + t0 + t)*rb,
                                                vq + (int64_t)t*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
                }
""")

edit("""            for (int g = 0; g < q_per_kv; g++)
                memcpy(ctx + ((int64_t)s*H + (kvh*q_per_kv + g))*hd,
                       cxg + (int64_t)g*kvd, (size_t)kvd*sizeof(float));
""",
     """            if (absorb)
                for (int g = 0; g < q_per_kv; g++)
                    coli_tq_unrotate(cxg + (size_t)g*kvd, kvd, COLI_TQ_SEED);
            for (int g = 0; g < q_per_kv; g++)
                memcpy(ctx + ((int64_t)s*H + (kvh*q_per_kv + g))*hd,
                       cxg + (int64_t)g*kvd, (size_t)kvd*sizeof(float));
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_tq_absorb: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".preabsorb"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_tq_absorb applied (%d edits, backup %s)" % (len(EDITS), bak))
