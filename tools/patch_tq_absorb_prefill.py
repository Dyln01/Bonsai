#!/usr/bin/env python3
"""patch_tq_absorb_prefill.py — Phase 3b-prefill: absorbed rotation in the flash tile loop.

Extends docs/ABSORB_ROT_QWEN36.md to prefill: with KV_TQ codec 1 + absorb on,
each query row of a flash block is rotated ONCE per (s0, kvh) block, K/V rows
dequantize to raw rotated-space levels (no per-row, per-block FWHT), scores dot
against rotated queries, and the online-softmax accumulators stay in rotated
space (rescales are linear) until a single unrotate per (row, head) at the
block's cx copy. KV8 and codec 0 untouched; Q36_TQ_ABSORB=0 restores verbatim.

Anchored exact-byte edits, house style. Backup qwen36.c.preabsp.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "qrot" in src:
    raise SystemExit("patch_tq_absorb_prefill: already applied")
if "g_q36_tq_absorb" not in src:
    raise SystemExit("patch_tq_absorb_prefill: patch_tq_absorb.py must be applied first")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# 1. rotate the block's query rows once per (s0, kvh)
edit("""                    int tmax = pos_base + s0 + rows;
""",
     """                    int tmax = pos_base + s0 + rows;
                    int absorb = m->kv_tq && m->tq_codec == 1 && g_q36_tq_absorb;  /* 3b-prefill */
                    float *qrot = absorb ? malloc((size_t)rows*q_per_kv*kvd*sizeof(float)) : NULL;
                    if (absorb && !qrot) { fprintf(stderr, "OOM qrot\\n"); exit(1); }
                    if (absorb)
                        for (int l = 0; l < rows; l++)
                            for (int g = 0; g < q_per_kv; g++)
                                coli_tq_rotate(query + ((int64_t)(s0+l)*H + kvh*q_per_kv + g)*hd,
                                               qrot + ((size_t)l*q_per_kv + g)*kvd, kvd, COLI_TQ_SEED);
""")

# 2. K dequant: rotated-space levels when absorbing
edit("""                            if (m->kv8 || m->kv_tq) {
                                q36_kv_dequant_noscale(Kl8 + (kbase + t0 + tt)*rb, kq + (size_t)tt*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
                                ksrc = kq + (size_t)tt*kvd;
                            } else ksrc = Kl_f + (kbase + t0 + tt)*kvd;
""",
     """                            if (m->kv8 || m->kv_tq) {
                                if (absorb) q36_q4_rot_noscale(Kl8 + (kbase + t0 + tt)*rb, kq + (size_t)tt*kvd, kvd);
                                else q36_kv_dequant_noscale(Kl8 + (kbase + t0 + tt)*rb, kq + (size_t)tt*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
                                ksrc = kq + (size_t)tt*kvd;
                            } else ksrc = Kl_f + (kbase + t0 + tt)*kvd;
""")

# 3. score dots against rotated queries
edit("""                                    sc[((size_t)l*q_per_kv + g)*TB + tt] = lim > 0
                                        ? dot_f32_lanes(query + ((int64_t)(s0+l)*H + hh)*hd, ksrc, kvd)
                                          * ((m->kv8 || m->kv_tq) ? Ksc[kbase + t0 + tt] * scale : scale)
                                        : -1e30f;
""",
     """                                    sc[((size_t)l*q_per_kv + g)*TB + tt] = lim > 0
                                        ? dot_f32_lanes(absorb ? qrot + ((size_t)l*q_per_kv + g)*kvd
                                                               : query + ((int64_t)(s0+l)*H + hh)*hd, ksrc, kvd)
                                          * ((m->kv8 || m->kv_tq) ? Ksc[kbase + t0 + tt] * scale : scale)
                                        : -1e30f;
""")

# 4. V dequant: rotated-space levels when absorbing
edit("""                        if (m->kv8 || m->kv_tq)
                            for (int tt = 0; tt < tn; tt++)
                                q36_kv_dequant_noscale(Vl8 + (kbase + t0 + tt)*rb, vq + (size_t)tt*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
""",
     """                        if (m->kv8 || m->kv_tq)
                            for (int tt = 0; tt < tn; tt++) {
                                if (absorb) q36_q4_rot_noscale(Vl8 + (kbase + t0 + tt)*rb, vq + (size_t)tt*kvd, kvd);
                                else q36_kv_dequant_noscale(Vl8 + (kbase + t0 + tt)*rb, vq + (size_t)tt*kvd, kvd, m->kv_tq, m->tq_bits, m->tq_codec);
                            }
""")

# 5. unrotate each accumulator once, before the cx copy
edit("""                        int li = l*q_per_kv + g, hh = kvh*q_per_kv + g;
                        float *cx = ctx + ((int64_t)(s0+l)*H + hh)*hd;
                        float inv = 1.f / lvs[li];
                        for (int d = 0; d < kvd; d++) cx[d] = acc[(size_t)li*kvd + d] * inv;
""",
     """                        int li = l*q_per_kv + g, hh = kvh*q_per_kv + g;
                        float *cx = ctx + ((int64_t)(s0+l)*H + hh)*hd;
                        float inv = 1.f / lvs[li];
                        if (absorb) coli_tq_unrotate(acc + (size_t)li*kvd, kvd, COLI_TQ_SEED);
                        for (int d = 0; d < kvd; d++) cx[d] = acc[(size_t)li*kvd + d] * inv;
""")

# 6. release the rotated-query block buffer
edit("""                    for (int l = 0; l < rows; l++) for (int g = 0; g < q_per_kv; g++) {
                        int li = l*q_per_kv + g, hh = kvh*q_per_kv + g;
                        float *cx = ctx + ((int64_t)(s0+l)*H + hh)*hd;
                        float inv = 1.f / lvs[li];
                        if (absorb) coli_tq_unrotate(acc + (size_t)li*kvd, kvd, COLI_TQ_SEED);
                        for (int d = 0; d < kvd; d++) cx[d] = acc[(size_t)li*kvd + d] * inv;
                    }
""",
     """                    for (int l = 0; l < rows; l++) for (int g = 0; g < q_per_kv; g++) {
                        int li = l*q_per_kv + g, hh = kvh*q_per_kv + g;
                        float *cx = ctx + ((int64_t)(s0+l)*H + hh)*hd;
                        float inv = 1.f / lvs[li];
                        if (absorb) coli_tq_unrotate(acc + (size_t)li*kvd, kvd, COLI_TQ_SEED);
                        for (int d = 0; d < kvd; d++) cx[d] = acc[(size_t)li*kvd + d] * inv;
                    }
                    free(qrot);
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_tq_absorb_prefill: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".preabsp"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_tq_absorb_prefill applied (%d edits, backup %s)" % (len(EDITS), bak))
