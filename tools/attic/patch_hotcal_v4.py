#!/usr/bin/env python3
"""patch_hotcal.py — TRUNK_MAP v4: calibration + per-layer fracs + warm-prefetch."""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode()
if "HOT_PROFILE" in src:
    print("already patched"); sys.exit(0)

def sub1(anchor, repl, tag):
    global src
    assert src.count(anchor) == 1, f"anchor {tag}: count={src.count(anchor)}"
    src = src.replace(anchor, repl)

# A+B: calibration + per-layer fracs
sub1("static int g_q36_ref_mode;",
"""static double q36_hot_frac(void);   /* defined below (v3) */
static int g_q36_ref_mode;
/* ---- HOT-NEURON v4: calibration ---------------------------------------- */
typedef struct { float *v; int n; } HotAcc;
static HotAcc g_hot_acc[512]; static int g_hot_acc_n = 0;
static char   g_hot_profile_path[1024] = {0};
static void q36_hot_acc_add(int layer, const float *g, const float *u, int n) {
    if (!g_hot_profile_path[0]) return;
    if (layer >= g_hot_acc_n) {
        int old = g_hot_acc_n; g_hot_acc_n = layer + 1;
        for (int i = old; i < g_hot_acc_n; i++) g_hot_acc[i] = (HotAcc){NULL, 0};
    }
    if (!g_hot_acc[layer].v) { g_hot_acc[layer].v = calloc((size_t)n, sizeof(float)); g_hot_acc[layer].n = n; }
    HotAcc *a = &g_hot_acc[layer];
    for (int i = 0; i < n; i++) {
        float sv = g[i];
        float act = (sv/(1.f+expf(-sv)))*u[i];
        a->v[i] += fabsf(act);
    }
}
static void q36_hot_profile_save(void) {
    if (!g_hot_profile_path[0] || !g_hot_acc_n) return;
    FILE *f = fopen(g_hot_profile_path, "w");
    if (!f) { fprintf(stderr, "[hot] cannot write %s\\n", g_hot_profile_path); return; }
    fprintf(f, "{\\n");
    for (int L = 0; L < g_hot_acc_n; L++) {
        if (!g_hot_acc[L].v) continue;
        fprintf(f, "  \\"L%d\\": {\\"n\\": %d, \\"energy\\": [", L, g_hot_acc[L].n);
        for (int i = 0; i < g_hot_acc[L].n; i++)
            fprintf(f, "%s%.6g", i ? "," : "", (double)g_hot_acc[L].v[i]);
        fprintf(f, "] }%s\\n", L+1 < g_hot_acc_n ? "," : "");
    }
    fprintf(f, "}\\n");
    fclose(f);
    fprintf(stderr, "[hot] profile written: %s (%d layers)\\n", g_hot_profile_path, g_hot_acc_n);
}
static double q36_layer_frac(int layer) {
    static double tbl[512]; static int tbl_ok = 0;
    if (!tbl_ok) {
        tbl_ok = 1;
        for (int i = 0; i < 512; i++) tbl[i] = -1.0;
        const char *p = getenv("HOT_LAYERS");
        if (p && *p) {
            FILE *f = fopen(p, "r");
            if (f) {
                static char buf[1 << 20];
                size_t rd = fread(buf, 1, sizeof(buf)-1, f); buf[rd] = 0; fclose(f);
                char *s = buf;
                while ((s = strstr(s, "\\"L")) != NULL) {
                    int ln = atoi(s+3);
                    char *c = strchr(s, ':');
                    if (!c) break;
                    double v = atof(c+1);
                    if (ln >= 0 && ln < 512 && v > 0 && v <= 1.0) tbl[ln] = v;
                    s = c + 1;
                }
                fprintf(stderr, "[hot] HOT_LAYERS loaded\\n");
            } else fprintf(stderr, "[hot] HOT_LAYERS: cannot open %s\\n", p);
        }
    }
    if (layer >= 0 && layer < 512 && tbl[layer] > 0) return tbl[layer];
    return q36_hot_frac();
}""", "cal-globals")

# hooks: stock S=1 silu + paged batch silu
sub1("for(int i=0;i<I;i++){float sv=g[i];g[i]=(sv/(1.f+expf(-sv)))*u[i];}",
"""for(int i=0;i<I;i++){float sv=g[i];g[i]=(sv/(1.f+expf(-sv)))*u[i];}
            { int _li = (int)(l - m->L); q36_hot_acc_add(_li, g, u, I); }""", "acc-stock")

sub1("for(int64_t q=0;q<(int64_t)rows*I;q++){float sv=bg[q];bg[q]=(sv/(1.f+expf(-sv)))*bu[q];}",
"""for(int64_t q=0;q<(int64_t)rows*I;q++){float sv=bg[q];bg[q]=(sv/(1.f+expf(-sv)))*bu[q];}
            { int _li = (int)(l - m->L); q36_hot_acc_add(_li, bg, bu, I); }""", "acc-paged")

sub1("    double hk = q36_hot_frac();\n    int Ik = (int)(I * hk);",
     "    double hk = q36_layer_frac((int)(l - m->L));\n    int Ik = (int)(I * hk);", "perlayer")

# profile env + save
sub1("    g_q36_ref_mode = is_ref;",
"""    g_q36_ref_mode = is_ref;
    { const char *hp = getenv("HOT_PROFILE"); if (hp && *hp) snprintf(g_hot_profile_path, sizeof g_hot_profile_path, "%s", hp); }""", "profile-env")

sub1("    q36_spec_free(&g_q36_run_spec);\n    tm_report();",
     "    q36_spec_free(&g_q36_run_spec);\n    q36_hot_profile_save();\n    tm_report();", "profile-save")

# C) warm-prefetch before generate()
sub1("    if (is_ref) { g_q36_oracle = full; g_q36_oracle_n = nfull; }   /* COLI_LOOKUP_FORCE */\n    generate(&m, prompt, np, n_new, out);",
"""    if (is_ref) { g_q36_oracle = full; g_q36_oracle_n = nfull; }   /* COLI_LOOKUP_FORCE */
    if (getenv("TRUNK_WARM") && getenv("TRUNK_WARM")[0]=='1') {
        double cover = q36_hot_frac();
        double w0 = now_s(); int64_t warmed = 0;
        for (int li = 0; li < m.c.n_layers; li++) {
            Layer *l = &m.L[li];
            const QW *ws[10] = { &l->q, &l->k, &l->v, &l->o, &l->sh_g, &l->sh_u, &l->sh_d, &l->dn_qkv, &l->dn_z, &l->dn_out };
            for (int wi = 0; wi < 10; wi++) {
                const QW *w = ws[wi];
                if (!w || !w->map) continue;
                int64_t bytes = w->map_len;
                if (cover < 1.0 && (wi == 4 || wi == 5 || wi == 6))
                    bytes = (int64_t)((double)bytes * cover) & ~(int64_t)4095;
                if (bytes <= 0) continue;
                void *base = (void*)((uintptr_t)w->map & ~(uintptr_t)4095);
                int64_t span = bytes + ((uintptr_t)w->map - (uintptr_t)base);
#ifdef MADV_WILLNEED
                madvise(base, span, MADV_WILLNEED);
#endif
                warmed += span;
            }
        }
        fprintf(stderr, "[warm] madvise(WILLNEED) over %.2f GB of mapped trunk (frac %.2f)\\n",
                warmed/1e9, cover);
    }
    generate(&m, prompt, np, n_new, out);""", "warm")

open(P, "w").write(src)
print("HOT-CAL v4 applied")
