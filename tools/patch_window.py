#!/usr/bin/env python3
"""patch_window.py — Phase 4b slice 1: per-layer resident neuron-row window.

Design contract: docs/WINDOWING_QWEN36.md. A FIFO cache per MLP-bearing layer
holding byte-identical packed int4 copies of (gate row, up row, down column)
for recently-used neuron indices, so the gather path in moe_ex answers hits
from RAM instead of re-reading scattered rows/columns from the mapped trunk
(the HDD/GRIFFIN value case). Determinism: copies are byte-identical to the
map, so every gemv sees the same bits whichever source answers — output is
independent of cache history (CI id-identity gate applies).

Knob Q36_WIN_MB (default 0 here; launcher policy + I/O gate are slice 3).
Int4-mapped dialect only; f16/resident containers skip the cache silently.
Report line [qwen36 win] at end of generate.

Anchored exact-byte edits, house style. Backup qwen36.c.prewin.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "Q36Win" in src:
    raise SystemExit("patch_window: already applied")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# 1. Layer field + knob global
edit("""    QW sh_g, sh_u, sh_d; float *sh_gate;   /* shared expert (dense, int8-during-load) + shared_expert_gate */
""",
     """    QW sh_g, sh_u, sh_d; float *sh_gate;   /* shared expert (dense, int8-during-load) + shared_expert_gate */
    struct Q36Win *win;                    /* Phase 4b: resident neuron-row window (NULL = off) */
""")

edit("""static int  g_q36_trunk_stream = 0;  /* 1 = TRUNK_MAP streaming mode (HDD/small-RAM) */
""",
     """static int  g_q36_trunk_stream = 0;  /* 1 = TRUNK_MAP streaming mode (HDD/small-RAM) */
static int  g_q36_win_mb = 0;        /* Q36_WIN_MB: resident neuron window budget (0 = off) */
static uint64_t g_win_hit, g_win_miss;
""")

edit("""    { const char *tv = getenv("Q36_TRUNK_STREAM"); if (tv && *tv) g_q36_trunk_stream = tv[0] == '1'; }
""",
     """    { const char *tv = getenv("Q36_TRUNK_STREAM"); if (tv && *tv) g_q36_trunk_stream = tv[0] == '1'; }
    { const char *wv = getenv("Q36_WIN_MB"); if (wv && *wv) g_q36_win_mb = atoi(wv); }
""")

# 2. window machinery, before moe_ex
edit("""static void qwen_shared_experts_cpu(Model *m, Layer *l, const float *x, int S,
""",
     """/* ---- Phase 4b: resident neuron-row window (byte-identical packed copies) ---- */
typedef struct Q36Win {
    int cap, tab, head, n;             /* FIFO slots, open-address table (pow2), fill */
    int *idx;                          /* slot -> neuron index (-1 empty) */
    int *slot_of;                      /* table: neuron -> slot (-1 empty) */
    uint8_t **grow, **urow, **dcol;    /* slot -> packed bytes (gate row, up row, down col) */
    float *gsc, *usc;                  /* slot -> row scale copies */
    int64_t rowb, dcolb;               /* packed bytes per gate/up row and per down column */
} Q36Win;

static int q36_win_tab_find(const Q36Win *w, int o) {
    unsigned h = (unsigned)o * 2654435761u;
    int i = (int)(h & (unsigned)(w->tab - 1));
    for (int probe = 0; probe < w->tab; probe++) {
        int s = w->slot_of[i];
        if (s < 0) return -1;
        if (w->idx[s] == o) return s;
        i = (i + 1) & (w->tab - 1);
    }
    return -1;
}

static void q36_win_tab_put(Q36Win *w, int o, int slot) {
    unsigned h = (unsigned)o * 2654435761u;
    int i = (int)(h & (unsigned)(w->tab - 1));
    for (int probe = 0; probe < w->tab; probe++) {
        int s = w->slot_of[i];
        if (s < 0 || w->idx[s] == o) { w->slot_of[i] = slot; return; }
        i = (i + 1) & (w->tab - 1);
    }
}

static void q36_win_ensure(Model *m, Layer *l, int D, int I) {
    if (l->win || g_q36_win_mb <= 0) return;
    if (!l->sh_g.map || l->sh_g.map_len != (int64_t)D * I / 2) return;   /* int4-mapped dialect only */
    if (!l->sh_d.map || l->sh_d.map_len != (int64_t)D * I / 2) return;
    int n_mlp = 0;
    for (int i = 0; i < m->c.n_layers; i++)
        if (m->L[i].sh_g.map && m->L[i].sh_g.map_len == (int64_t)D * I / 2) n_mlp++;
    if (n_mlp < 1) return;
    int64_t per = (int64_t)(D / 2) * 2 + (int64_t)(D / 2);                /* gate+up rows + down col */
    int64_t cap = ((int64_t)g_q36_win_mb * 1024 * 1024 / n_mlp) / (per + 16);
    if (cap < 64) return;
    Q36Win *w = calloc(1, sizeof *w);
    int tab = 64; while (tab < cap * 2) tab <<= 1;
    w->cap = (int)cap; w->tab = tab; w->rowb = D / 2; w->dcolb = D / 2;
    w->idx = malloc((size_t)cap * sizeof(int));
    w->slot_of = malloc((size_t)tab * sizeof(int));
    w->grow = calloc((size_t)cap, sizeof(uint8_t *));
    w->urow = calloc((size_t)cap, sizeof(uint8_t *));
    w->dcol = calloc((size_t)cap, sizeof(uint8_t *));
    w->gsc = malloc((size_t)cap * sizeof(float));
    w->usc = malloc((size_t)cap * sizeof(float));
    if (!w->idx || !w->slot_of || !w->grow || !w->urow || !w->dcol || !w->gsc || !w->usc) { free(w); return; }
    for (int i = 0; i < cap; i++) w->idx[i] = -1;
    for (int i = 0; i < tab; i++) w->slot_of[i] = -1;
    for (int i = 0; i < cap; i++) {
        w->grow[i] = malloc((size_t)(D / 2)); w->urow[i] = malloc((size_t)(D / 2)); w->dcol[i] = malloc((size_t)(D / 2));
        if (!w->grow[i] || !w->urow[i] || !w->dcol[i]) { free(w); l->win = NULL; return; }
    }
    l->win = w;
}

static int q36_win_lookup(Model *m, Layer *l, int o) {
    if (!l->win) return -1;
    int s = q36_win_tab_find(l->win, o);
    if (s >= 0) g_win_hit++; else g_win_miss++;
    return s;
}

static void q36_win_insert(Model *m, Layer *l, int o, int D, int I) {
    Q36Win *w = l->win;
    if (!w || q36_win_tab_find(w, o) >= 0) return;
    int s = w->head; w->head = (w->head + 1) % w->cap;
    if (w->idx[s] >= 0) {   /* evict: drop its table entry */
        unsigned h = (unsigned)w->idx[s] * 2654435761u;
        int i = (int)(h & (unsigned)(w->tab - 1));
        for (int probe = 0; probe < w->tab; probe++) {
            if (w->slot_of[i] == s) { w->slot_of[i] = -1; break; }
            i = (i + 1) & (w->tab - 1);
        }
    }
    const uint8_t *gm = (const uint8_t *)l->sh_g.map, *um = (const uint8_t *)l->sh_u.map, *dm = (const uint8_t *)l->sh_d.map;
    memcpy(w->grow[s], gm + (int64_t)o * w->rowb, (size_t)w->rowb);
    memcpy(w->urow[s], um + (int64_t)o * w->rowb, (size_t)w->rowb);
    for (int d = 0; d < D; d += 2) {   /* down column o: nibbles (d,o) for all d, packed pair-wise */
        w->dcol[s][d / 2] = 0;         /* filled below, nibble at a time */
    }
    for (int d = 0; d < D; d++) {
        uint8_t b = dm[(int64_t)d * (I / 2) + (o >> 1)];
        int nib = (o & 1) ? (b >> 4) & 0xF : b & 0xF;
        if (d & 1) w->dcol[s][d / 2] |= (uint8_t)(nib << 4);
        else       w->dcol[s][d / 2] |= (uint8_t)nib;
    }
    w->gsc[s] = l->sh_g.sc ? l->sh_g.sc[o] : 1.0f;
    w->usc[s] = l->sh_u.sc ? l->sh_u.sc[o] : 1.0f;
    w->idx[s] = o;
    q36_win_tab_put(w, o, s);
    if (w->n < w->cap) w->n++;
}

static void qwen_shared_experts_cpu(Model *m, Layer *l, const float *x, int S,
""")

# 3. gate/up row views answer from the window when present
edit("""        for (int j = 0; j < K; j++) {
            int o = ix[j];
            /* one row of sh_g and sh_u at output index o, dot with xs (width D) */
            QW ng = l->sh_g; ng.O = 1; ng.I = D;
            /* shift the map to row o: rows are o*(I/2 or I) bytes — build a row view */
            float gval, uval;
            {
                const QW *w = &l->sh_g;
                if (w->map) {
""",
     """        for (int j = 0; j < K; j++) {
            int o = ix[j];
            /* one row of sh_g and sh_u at output index o, dot with xs (width D) */
            QW ng = l->sh_g; ng.O = 1; ng.I = D;
            /* shift the map to row o: rows are o*(I/2 or I) bytes — build a row view */
            float gval, uval;
            int wslot = q36_win_lookup(m, l, o);
            if (wslot < 0 && l->win) q36_win_insert(m, l, o, D, I), wslot = q36_win_tab_find(l->win, o);
            {
                const QW *w = &l->sh_g;
                if (w->map && wslot >= 0) {
                    QW rv = *w; rv.map = (char *)l->win->grow[wslot]; rv.map_len = l->win->rowb;
                    rv.O = 1; rv.I = D; rv.sc = &l->win->gsc[wslot];
                    matmul_d(&gval, xs, &rv, 1, D, 1);
                } else if (w->map) {
""")

edit("""            {
                const QW *w = &l->sh_u;
                if (w->map) {
                    int64_t rowbytes = (w->map_len == (int64_t)D*I/2) ? (D/2) : (D * 2);
                    QW rv = *w; rv.map = (char*)w->map + (int64_t)o * rowbytes; rv.map_len = rowbytes; rv.O = 1; rv.I = D; rv.sc = w->sc ? w->sc + o : NULL;
                    matmul_d(&uval, xs, &rv, 1, D, 1);
                } else {
""",
     """            {
                const QW *w = &l->sh_u;
                if (w->map && wslot >= 0) {
                    QW rv = *w; rv.map = (char *)l->win->urow[wslot]; rv.map_len = l->win->rowb;
                    rv.O = 1; rv.I = D; rv.sc = &l->win->usc[wslot];
                    matmul_d(&uval, xs, &rv, 1, D, 1);
                } else if (w->map) {
                    int64_t rowbytes = (w->map_len == (int64_t)D*I/2) ? (D/2) : (D * 2);
                    QW rv = *w; rv.map = (char*)w->map + (int64_t)o * rowbytes; rv.map_len = rowbytes; rv.O = 1; rv.I = D; rv.sc = w->sc ? w->sc + o : NULL;
                    matmul_d(&uval, xs, &rv, 1, D, 1);
                } else {
""")

# 4. ensure+init at the top of the gather path, and the down-column hook
edit("""        /* gather-selected neurons: compute gate/up rows gs->idx[0..k), combine,
         * then down over the same original indices. */
        int K = gs->k;
""",
     """        /* gather-selected neurons: compute gate/up rows gs->idx[0..k), combine,
         * then down over the same original indices. */
        q36_win_ensure(m, l, D, I);
        int K = gs->k;
""")

edit("""            if (dw->map && dw->map_len == (int64_t)D*I/2) {
                const uint8_t *mp = (const uint8_t*)dw->map; const float *sc = dw->sc;
                #pragma omp parallel for schedule(static)
                for (int d = 0; d < D; d++) {
                    const uint8_t *row = mp + (int64_t)d * (I >> 1);
                    float acc = 0;
                    for (int j = 0; j < K; j++) {
                        int i2 = ix[j];
                        uint8_t b = row[i2 >> 1];
""",
     """            if (dw->map && dw->map_len == (int64_t)D*I/2) {
                const uint8_t *mp = (const uint8_t*)dw->map; const float *sc = dw->sc;
                /* window: cached down columns answer from RAM (sequential per d) */
                const uint8_t **csrc = NULL;
                if (l->win) {
                    csrc = malloc((size_t)K * sizeof(uint8_t *));
                    for (int j = 0; j < K; j++) {
                        int s2 = q36_win_tab_find(l->win, ix[j]);
                        csrc[j] = s2 >= 0 ? l->win->dcol[s2] : NULL;
                    }
                }
                #pragma omp parallel for schedule(static)
                for (int d = 0; d < D; d++) {
                    const uint8_t *row = mp + (int64_t)d * (I >> 1);
                    float acc = 0;
                    for (int j = 0; j < K; j++) {
                        int i2 = ix[j];
                        uint8_t b = csrc && csrc[j] ? csrc[j][d >> 1] : row[i2 >> 1];
                        int nibsel = (csrc && csrc[j]) ? (d & 1) : (i2 & 1);
                        int8_t e = (int8_t)((nibsel) ? (b >> 4) : (b & 0xF));
                        if (e & 8) e -= 16;
                        acc += (float)e * gg[j];
                    }
                    hh[d] = acc * (sc ? sc[d] : 1.0f);
                }
                free((void*)csrc);
                if (0) {
                #pragma omp parallel for schedule(static)
                for (int d = 0; d < D; d++) {
                    const uint8_t *row = mp + (int64_t)d * (I >> 1);
                    float acc = 0;
                    for (int j = 0; j < K; j++) {
                        int i2 = ix[j];
                        uint8_t b = row[i2 >> 1];
""")

# close the dead duplicate branch after its original body
edit("""                        if (e & 8) e -= 16;
                        acc += (float)e * gg[j];
                    }
                    hh[d] = acc * (sc ? sc[d] : 1.0f);
                }
            } else if (dw->map) {
""",
     """                        if (e & 8) e -= 16;
                        acc += (float)e * gg[j];
                    }
                    hh[d] = acc * (sc ? sc[d] : 1.0f);
                }
                }
            } else if (dw->map) {
""")

# 5. report line at end of generate
edit("""    q36_spec_end(m, sp);
""",
     """    if (g_q36_win_mb > 0)
        fprintf(stderr, "[qwen36 win] hits %llu misses %llu (%.1f%% hit-rate), budget %d MB\\n",
                (unsigned long long)g_win_hit, (unsigned long long)g_win_miss,
                (g_win_hit + g_win_miss) ? 100.0 * g_win_hit / (g_win_hit + g_win_miss) : 0.0,
                g_q36_win_mb);
    q36_spec_end(m, sp);
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_window: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".prewin"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_window applied (%d edits, backup %s)" % (len(EDITS), bak))
