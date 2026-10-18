#!/usr/bin/env python3
"""patch_mtp.py — Phase 2b part 1: MTP head LOADER + knobs for qwen36.c.

Part 1 scope (design contract: docs/MTP_DESIGN_QWEN36.md):
  - meta keys mtp/mtp_layers in the Cfg parse (qwen36_meta.json)
  - Q36_MTP=1 / Q36_MTP_DRAFTS=0..3 / Q36_MTP_FORCE=<str> env knobs
  - q36_mtp_attach(): loads the DENSE head (norms + qk-normed full attention +
    mlp.{gate,up,down}) from model-mtp.safetensors — the file is already in
    m->S because st_scan_dir takes every *.safetensors in the container
  - refuse-gates: knob w/o meta, meta w/o tensors, MoE head (part 2 territory)
  - report line [qwen36 MTP]; q36_mtp_free() in the segment destroy path
Drafting/verify/force-oracle (the SPEC_SRC_MTP proposal beside lookup) is
part 2; with part 1 the head loads and stays inert, so knob-off runs are
byte-identical and knob-on runs differ only by the report line (CI gate).

Every edit is an exact-byte anchored replacement (count==1 asserted), house
style. Backs up to qwen36.c.premtp first. Refuses to run twice.

Usage: python3 patch_mtp.py <path-to-qwen36.c>
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "Q36MTPHead" in src:
    raise SystemExit("patch_mtp: already applied (Q36MTPHead present)")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# 1. Model field (opaque pointer: the typedef lands later, before load_meta)
edit("""    QW lm_head;
""",
     """    QW lm_head;
    struct Q36MTPHead *mtp;   /* Phase 2b: MTP draft head (NULL until attach) */
""")

# 2. Cfg fields
edit("""    int expert_down_bits, expert_down_gs;
""",
     """    int expert_down_bits, expert_down_gs;
    int mtp, mtp_layers;      /* Phase 2b: qwen36_meta.json "mtp"/"mtp_layers" */
""")

# 3. meta parse
edit("""        G("expert_gs", expert_gs);
""",
     """        G("expert_gs", expert_gs);
        G("mtp", mtp); G("mtp_layers", mtp_layers);
""")

# 4. head struct + knobs + loader/free, before load_meta
edit("""static void load_meta(Cfg *c, const char *snap) {
""",
     """/* ================= Phase 2b: MTP head (part 1: loader + knobs) =========
 * The head's tensors ride in m->S already: st_scan_dir takes every
 * *.safetensors in the container, so model-mtp.safetensors needs no second
 * shards instance — and an MTP-unaware engine never requests mtp.* names,
 * which is exactly the stock-safety property measured in Phase 2a. */
static int  g_q36_mtp_on = 0;
static int  g_q36_mtp_drafts = 2;
static char g_q36_mtp_force[32] = {0};

typedef struct Q36MTPHead {
    int loaded, dense;
    int H, I;
    float *fc, *n_emb, *n_hid, *n_in, *n_post, *n_q, *n_k, *n_fin;
    float *qw, *kw, *vw, *ow, *gw, *uw, *dw;
    int64_t n_fc, n_qw, n_kw, n_vw, n_ow, n_gw, n_uw, n_dw;
} MTPHead;

static float *mtp_f32(shards *S, const char *name, int64_t *numel) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "[qwen36 MTP] missing tensor %s\\n", name); exit(1); }
    float *p = malloc((size_t)t->numel * sizeof(float));
    if (!p) { fprintf(stderr, "[qwen36 MTP] OOM on %s\\n", name); exit(1); }
    int64_t n = st_read_f32(S, name, p, 0);
    if (n != t->numel) { fprintf(stderr, "[qwen36 MTP] short read %s\\n", name); exit(1); }
    if (numel) *numel = n;
    return p;
}

static void q36_mtp_attach(Model *m) {
    if (!g_q36_mtp_on) return;                    /* knob off: fully inert */
    if (!m->c.mtp) {
        fprintf(stderr, "[qwen36 MTP] Q36_MTP=1 but qwen36_meta.json has no \\"mtp\\" key — refusing\\n");
        exit(1);
    }
    if (!st_find(&m->S, "mtp.fc.weight")) {
        fprintf(stderr, "[qwen36 MTP] meta.mtp set but no mtp.* tensors in the container — refusing\\n");
        exit(1);
    }
    MTPHead *h = calloc(1, sizeof *h);
    if (!h) { fprintf(stderr, "[qwen36 MTP] OOM\\n"); exit(1); }
    h->H = m->c.hidden;
    h->fc    = mtp_f32(&m->S, "mtp.fc.weight", &h->n_fc);
    h->n_emb = mtp_f32(&m->S, "mtp.pre_fc_norm_embedding.weight", NULL);
    h->n_hid = mtp_f32(&m->S, "mtp.pre_fc_norm_hidden.weight", NULL);
    h->n_fin = mtp_f32(&m->S, "mtp.norm.weight", NULL);
    h->n_in  = mtp_f32(&m->S, "mtp.layers.0.input_layernorm.weight", NULL);
    h->n_post= mtp_f32(&m->S, "mtp.layers.0.post_attention_layernorm.weight", NULL);
    {
        char nm[256]; const char *p0 = "mtp.layers.0.self_attn.";
        snprintf(nm, sizeof nm, "%sq_proj.weight", p0); h->qw = mtp_f32(&m->S, nm, &h->n_qw);
        snprintf(nm, sizeof nm, "%sk_proj.weight", p0); h->kw = mtp_f32(&m->S, nm, &h->n_kw);
        snprintf(nm, sizeof nm, "%sv_proj.weight", p0); h->vw = mtp_f32(&m->S, nm, &h->n_vw);
        snprintf(nm, sizeof nm, "%so_proj.weight", p0); h->ow = mtp_f32(&m->S, nm, &h->n_ow);
        if (m->c.has_qk_norm) {
            snprintf(nm, sizeof nm, "%sq_norm.weight", p0); h->n_q = mtp_f32(&m->S, nm, NULL);
            snprintf(nm, sizeof nm, "%sk_norm.weight", p0); h->n_k = mtp_f32(&m->S, nm, NULL);
        }
        /* part 1 loads the DENSE head (mlp. prefix); the MoE head (router +
         * fused experts, resident int4) lands with drafting in part 2 */
        if (st_find(&m->S, "mtp.layers.0.mlp.gate_proj.weight")) {
            h->dense = 1;
            h->gw = mtp_f32(&m->S, "mtp.layers.0.mlp.gate_proj.weight", &h->n_gw);
            h->uw = mtp_f32(&m->S, "mtp.layers.0.mlp.up_proj.weight",   &h->n_uw);
            h->dw = mtp_f32(&m->S, "mtp.layers.0.mlp.down_proj.weight", &h->n_dw);
            h->I = (int)(h->n_uw / h->H);
        } else {
            fprintf(stderr, "[qwen36 MTP] MoE head in container; part 1 loads dense heads only (part 2 adds experts) — refusing\\n");
            exit(1);
        }
    }
    h->loaded = 1;
    m->mtp = h;
    fprintf(stderr, "[qwen36 MTP] head loaded: %d layer(s), dense=%d, hidden=%d, ~%ld MB resident (drafts=%d force=%s)\\n",
            m->c.mtp_layers > 0 ? m->c.mtp_layers : 1, h->dense, h->H,
            (long)((h->n_fc + h->n_qw + h->n_kw + h->n_vw + h->n_ow +
                    h->n_gw + h->n_uw + h->n_dw) * 4 / (1024 * 1024)),
            g_q36_mtp_drafts, g_q36_mtp_force[0] ? g_q36_mtp_force : "-");
}

static void q36_mtp_free(Model *m) {
    MTPHead *h = m->mtp;
    if (!h) return;
    free(h->fc); free(h->n_emb); free(h->n_hid); free(h->n_in); free(h->n_post);
    free(h->n_q); free(h->n_k); free(h->n_fin);
    free(h->qw); free(h->kw); free(h->vw); free(h->ow);
    free(h->gw); free(h->uw); free(h->dw);
    free(h); m->mtp = NULL;
}

static void load_meta(Cfg *c, const char *snap) {
""")

# 5. env knobs, beside the HOT_PROFILE parse
edit("""    { const char *hp = getenv("HOT_PROFILE"); if (hp && *hp) snprintf(g_hot_profile_path, sizeof g_hot_profile_path, "%s", hp); }
""",
     """    { const char *hp = getenv("HOT_PROFILE"); if (hp && *hp) snprintf(g_hot_profile_path, sizeof g_hot_profile_path, "%s", hp); }
    { const char *mv = getenv("Q36_MTP"); if (mv && mv[0] == '1') g_q36_mtp_on = 1;
      const char *md = getenv("Q36_MTP_DRAFTS");
      if (md && *md) { g_q36_mtp_drafts = atoi(md);
                       if (g_q36_mtp_drafts < 0 || g_q36_mtp_drafts > 3) {
                           fprintf(stderr, "Q36_MTP_DRAFTS must be 0..3\\n"); exit(1); } }
      const char *mf = getenv("Q36_MTP_FORCE");
      if (mf && *mf) snprintf(g_q36_mtp_force, sizeof g_q36_mtp_force, "%s", mf); }
""")

# 6. attach right after the shards open (st_find needs m->S)
edit("""    st_init(&m->S, snap);
    Cfg *c = &m->c;
""",
     """    st_init(&m->S, snap);
    q36_mtp_attach(m);
    Cfg *c = &m->c;
""")

# 7. free in the segment destroy path (NULL-guarded: double-destroy safe)
edit("""    if (!engine) return;
    Model *model = &engine->model;
""",
     """    if (!engine) return;
    Model *model = &engine->model;
    q36_mtp_free(model);
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_mtp: anchor count=%d (want 1): %r" % (src.count(old), old[:60]))
    src = src.replace(old, new, 1)

bak = P + ".premtp"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mtp part 1 applied (%d edits, backup %s)" % (len(EDITS), bak))
