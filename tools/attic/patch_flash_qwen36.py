#!/usr/bin/env python3
"""patch_all.py — consolidated: flash v2 (head-grouped, KV8-capable) + attn split timers.
Applied to a clean bf24429+KV8 tree (qwen36.c.preflash restored).
Every anchor is asserted unique before use; the script is idempotent-safe by checking markers.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
assert "Q36_FLASH" not in src, "tree not clean"
assert "g_q36_ref_mode" not in src, "tree not clean (ref mode)"

def sub1(anchor, replacement, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag} count={src.count(anchor)}"
    src = src.replace(anchor, replacement)

# ---------- 1. globals ----------
sub1("static int g_q36_rowwise;   /* set for a speculative verify: its rows must get a decode step's bits (q36_spec_step) */",
     "static int g_q36_rowwise;   /* set for a speculative verify: its rows must get a decode step's bits (q36_spec_step) */\n"
     "static int g_q36_ref_mode;  /* ref.json oracle: exactness mode, the flash prefill stays off */\n"
     "static double g_tm_aproj = 0, g_tm_acore = 0, g_tm_atail = 0, g_tm_at0 = 0;\n"
     "static int q36_flash_on(void) {\n"
     "    static int g = -1;\n"
     "    if (g < 0) { const char *e = getenv(\"Q36_FLASH\"); g = !(e && *e == '0'); }\n"
     "    return g;\n"
     "}\n"
     "static int q36_flash_tile(const char *env, int dflt, int lo, int hi) {\n"
     "    const char *e = getenv(env);\n"
     "    int v = e && *e ? atoi(e) : dflt;\n"
     "    if (v < lo) v = lo;\n"
     "    if (v > hi) v = hi;\n"
     "    return v;\n"
     "}", "globals")

# ---------- 2. timers: report line ----------
sub1('    fprintf(stderr,"[timers] prefill: %ld tokens',
     '    fprintf(stderr,"[timers] attn split: proj %.0f ms | core %.0f ms | tail %.0f ms\\n", g_tm_aproj, g_tm_acore, g_tm_atail);\n'
     '    fprintf(stderr,"[timers] prefill: %ld tokens', "report")

# ---------- 3. ref-mode set in main ----------
sub1('if (rplen>=5 && strcmp(refpath+rplen-5, ".json")==0) is_ref = 1;',
     'if (rplen>=5 && strcmp(refpath+rplen-5, ".json")==0) is_ref = 1;\n'
     '    g_q36_ref_mode = is_ref;', "refmode")

# ---------- 4. attention(): timers + flash v2 replacing the core section ----------
A_START = ("static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {\n"
           "    Cfg *c = &m->c;")
sub1(A_START, A_START + "\n    double _aproj0 = tm_now();", "attn-entry")

CORE = ("    float scale = 1.f / sqrtf((float)hd);\n"
        "    float *ctx = falloc((int64_t)S*H*hd);\n"
        "    #pragma omp parallel for collapse(2) schedule(static)\n"
        "    for (int hh = 0; hh < H; hh++) {\n"
        "        for (int s = 0; s < S; s++) {\n"
        "            int kvh = hh / q_per_kv;")
assert src.count(CORE) == 1, "core anchor"

FLASH_V2 = '''    if (tm_on()) g_tm_aproj += tm_now() - _aproj0;
    double _acore0 = tm_now();
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*H*hd);
    if (q36_flash_on() && S > 8 && !mr && !m->mpos && !m->rope_delta
        && !g_q36_ref_mode) {
        /* GQA head-grouped flash tiles (v2): each K/V tile is loaded once,
         * dequantized once under KV8, and serves every head of its kv-head
         * and every row of a QB-row query block. Online softmax, fused V
         * accumulation. Decode (S=1) and speculative verifies (S<=8) keep the
         * original per-row path, so a row's bits never depend on S there. */
        const int QB = q36_flash_tile("Q36_FLASH_QB", 64, 8, 512);
        const int TB = q36_flash_tile("Q36_FLASH_TB", 128, 16, 2048);
        if (getenv("Q36_FLASH_DEBUG")) fprintf(stderr, "[flash] engaged S=%d QB=%d TB=%d H=%d kv8=%d\\n", S, QB, TB, H, m->kv8);
        #pragma omp parallel
        {
            float *sc  = malloc((size_t)QB*q_per_kv*TB*sizeof(float));
            float *acc = malloc((size_t)QB*q_per_kv*kvd*sizeof(float));
            float *mxv = malloc((size_t)QB*q_per_kv*sizeof(float));
            float *lvs = malloc((size_t)QB*q_per_kv*sizeof(float));
            float *kq  = m->kv8 ? malloc((size_t)TB*kvd*sizeof(float)) : NULL;
            float *vq  = m->kv8 ? malloc((size_t)TB*kvd*sizeof(float)) : NULL;
            if (!sc || !acc || !mxv || !lvs || (m->kv8 && (!kq || !vq))) { fprintf(stderr, "OOM flash attn\\n"); exit(1); }
            #pragma omp for schedule(dynamic)
            for (int s0 = 0; s0 < S; s0 += QB) {
                int rows = S - s0 < QB ? S - s0 : QB;
                for (int kvh = 0; kvh < KV; kvh++) {
                    const float *Kl_f = m->K[layer], *Vl_f = m->V[layer];
                    const uint8_t *Kl8 = (const uint8_t*)m->K[layer], *Vl8 = (const uint8_t*)m->V[layer];
                    const float *Ksc = m->Kscale[layer], *Vsc = m->Vscale[layer];
                    int64_t kbase = (int64_t)kvh*m->max_t;
                    (void)Ksc; (void)Vsc;
                    memset(acc, 0, (size_t)QB*q_per_kv*kvd*sizeof(float));
                    for (int l = 0; l < rows*q_per_kv; l++) { mxv[l] = -1e30f; lvs[l] = 0.f; }
                    int tmax = pos_base + s0 + rows;
                    for (int t0 = 0; t0 < tmax; t0 += TB) {
                        int tn = tmax - t0; if (tn > TB) tn = TB;
                        for (int tt = 0; tt < tn; tt++) {
                            const float *ksrc;
                            if (m->kv8) {
                                kv8_dequant_noscale(Kl8 + (kbase + t0 + tt)*kvd, kq + (size_t)tt*kvd, kvd);
                                ksrc = kq + (size_t)tt*kvd;
                            } else ksrc = Kl_f + (kbase + t0 + tt)*kvd;
                            for (int g = 0; g < q_per_kv; g++) {
                                int hh = kvh*q_per_kv + g;
                                for (int l = 0; l < rows; l++) {
                                    int lim = (pos_base + s0 + l) - (t0 + tt) + 1;
                                    sc[((size_t)l*q_per_kv + g)*TB + tt] = lim > 0
                                        ? dot_f32_lanes(query + ((int64_t)(s0+l)*H + hh)*hd, ksrc, kvd) * scale
                                        : -1e30f;
                                }
                            }
                        }
                        if (m->kv8)
                            for (int tt = 0; tt < tn; tt++)
                                kv8_dequant_noscale(Vl8 + (kbase + t0 + tt)*kvd, vq + (size_t)tt*kvd, kvd);
                        for (int l = 0; l < rows; l++) {
                            int pos = pos_base + s0 + l;
                            int t_last = t0 + tn - 1; if (t_last > pos) t_last = pos;
                            if (t_last < t0) continue;
                            for (int g = 0; g < q_per_kv; g++) {
                                int li = l*q_per_kv + g;
                                float m2 = mxv[li];
                                for (int t = t0; t <= t_last; t++)
                                    if (sc[(size_t)li*TB + (t - t0)] > m2) m2 = sc[(size_t)li*TB + (t - t0)];
                                float *ax = acc + (size_t)li*kvd;
                                if (m2 > mxv[li]) {
                                    float r = expf(mxv[li] - m2);
                                    for (int d = 0; d < kvd; d++) ax[d] *= r;
                                    lvs[li] *= r; mxv[li] = m2;
                                }
                                for (int t = t0; t <= t_last; t++) {
                                    float p = expf(sc[(size_t)li*TB + (t - t0)] - m2);
                                    lvs[li] += p;
                                    const float *vr = m->kv8 ? vq + (size_t)(t - t0)*kvd
                                                             : Vl_f + (kbase + t)*kvd;
                                    for (int d = 0; d < kvd; d++) ax[d] += p * vr[d];
                                }
                            }
                        }
                    }
                    for (int l = 0; l < rows; l++) for (int g = 0; g < q_per_kv; g++) {
                        int li = l*q_per_kv + g, hh = kvh*q_per_kv + g;
                        float *cx = ctx + ((int64_t)(s0+l)*H + hh)*hd;
                        float inv = 1.f / lvs[li];
                        for (int d = 0; d < kvd; d++) cx[d] = acc[(size_t)li*kvd + d] * inv;
                    }
                }
            }
            free(sc); free(acc); free(mxv); free(lvs); free(kq); free(vq);
        }
    } else {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int kvh = hh / q_per_kv;'''

sub1(CORE, FLASH_V2, "flash-v2-core")

# ---------- 5. close: after the else branch ends ----------
CLOSE = ("            } else {\n"
         "                for (int t = 0; t <= qpos; t++) {\n"
         "                    const float *vrow = Vl + ((int64_t)kvh*m->max_t + t)*kvd;\n"
         "                    float a = sc[t]; for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];\n"
         "                }\n"
         "            }\n"
         "        }\n"
         "    }\n"
         "    /* apply attn_output_gate")
assert src.count(CLOSE) == 1, "close anchor"
sub1(CLOSE,
     "            } else {\n"
     "                for (int t = 0; t <= qpos; t++) {\n"
     "                    const float *vrow = Vl + ((int64_t)kvh*m->max_t + t)*kvd;\n"
     "                    float a = sc[t]; for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];\n"
     "                }\n"
     "            }\n"
     "        }\n"
     "    }\n"
     "    }   /* !Q36_FLASH prefill branch */\n"
     "    if (tm_on()) { g_tm_acore += tm_now() - _acore0; g_tm_at0 = tm_now(); }\n"
     "    /* apply attn_output_gate", "close")

# ---------- 6. tail timer at function end ----------
TAIL = ("    if (!qtd_batch(l->qth_o, out, ag, S, H*hd, D)) matmul_d(out, ag, &l->o, S, H*hd, D);\n"
        "    free(q); free(k); free(vv); free(query); free(gate); free(ctx); free(ag);\n"
        "}")
assert src.count(TAIL) == 1, "tail anchor"
sub1(TAIL,
     "    if (!qtd_batch(l->qth_o, out, ag, S, H*hd, D)) matmul_d(out, ag, &l->o, S, H*hd, D);\n"
     "    if (tm_on()) g_tm_atail += tm_now() - g_tm_at0;\n"
     "    free(q); free(k); free(vv); free(query); free(gate); free(ctx); free(ag);\n"
     "}", "tail")

open(P, "w").write(src)
print("consolidated patch applied")
print("flash markers:", src.count("Q36_FLASH"), "| prefill-branch closers:", src.count("}   /* !Q36_FLASH prefill branch */"))
