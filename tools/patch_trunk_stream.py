#!/usr/bin/env python3
"""patch_trunk_stream.py — HDD/small-RAM streaming mode for TRUNK_MAP.

On boxes whose RAM cannot hold the trunk AND whose disk is rotational, the
default TRUNK_MAP behaviour livelocks mid-prefill: layer i faults in while
madvise(WILLNEED) streams layer i+2 and kernel readahead adds a third stream
(seek-bound random I/O on an HDD), while the page cache — which can never hold
the trunk — pushes direct reclaim into thrash. Observed in the field as
"calibrate frozen at layer 48 for hours" on an 8 GB mini-PC with an HDD;
invisible on NVMe dev boxes and on synthetic containers.

Q36_TRUNK_STREAM=1 (default ON whenever HOT_PROFILE is set, i.e. calibrate):
  - no greedy i+2 WILLNEED prefetch (sequential faulting only)
  - MADV_DONTNEED each layer's ten mapped slabs right after its MoE add, so the
    cache footprint stays ~one layer instead of the whole trunk: reclaim has
    nothing to scan, per-layer time becomes steady slab_read/BW + compute
Q36_TRUNK_STREAM=0 restores today's behaviour (NVMe boxes with RAM to spare).

Anchored exact-byte edits, house style. Backup qwen36.c.prestream.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "g_q36_trunk_stream" in src:
    raise SystemExit("patch_trunk_stream: already applied")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

edit("""static int  g_q36_progress = 1;   /* Q36_PROGRESS=0 disables prefill progress lines */
""",
     """static int  g_q36_progress = 1;   /* Q36_PROGRESS=0 disables prefill progress lines */
static int  g_q36_trunk_stream = 0;  /* 1 = TRUNK_MAP streaming mode (HDD/small-RAM) */
""")

edit("""    { const char *hp = getenv("HOT_PROFILE"); if (hp && *hp) snprintf(g_hot_profile_path, sizeof g_hot_profile_path, "%s", hp); }
""",
     """    { const char *hp = getenv("HOT_PROFILE"); if (hp && *hp) snprintf(g_hot_profile_path, sizeof g_hot_profile_path, "%s", hp); }
    g_q36_trunk_stream = g_hot_profile_path[0] ? 1 : 0;   /* calibrate streams by default */
    { const char *tv = getenv("Q36_TRUNK_STREAM"); if (tv && *tv) g_q36_trunk_stream = tv[0] == '1'; }
""")

edit("""        if (i + 2 < c->n_layers && m->L[i + 2].q.map) {   /* TRUNK_MAP readahead */
            const QW *nq[6] = { &m->L[i+2].q, &m->L[i+2].k, &m->L[i+2].v, &m->L[i+2].o,
                                &m->L[i+2].sh_g, &m->L[i+2].sh_u };
            for (int qw = 0; qw < 6; qw++)
                if (nq[qw]->map) madvise((void*)(((uintptr_t)nq[qw]->map) & ~(uintptr_t)4095),
                                         (size_t)nq[qw]->map_len + 4096, MADV_WILLNEED);
        }
""",
     """        if (!g_q36_trunk_stream && i + 2 < c->n_layers && m->L[i + 2].q.map) {   /* TRUNK_MAP readahead */
            const QW *nq[6] = { &m->L[i+2].q, &m->L[i+2].k, &m->L[i+2].v, &m->L[i+2].o,
                                &m->L[i+2].sh_g, &m->L[i+2].sh_u };
            for (int qw = 0; qw < 6; qw++)
                if (nq[qw]->map) madvise((void*)(((uintptr_t)nq[qw]->map) & ~(uintptr_t)4095),
                                         (size_t)nq[qw]->map_len + 4096, MADV_WILLNEED);
        }
""")

edit("""        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);
        if (allow_prefetch && g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
""",
     """        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);
#ifdef MADV_WILLNEED
        if (g_q36_trunk_stream) {   /* streaming: this layer's slabs are done — keep the
                                       cache footprint at ~one layer, not the whole trunk,
                                       so reclaim has nothing to thrash on small boxes */
            const QW *cq[10] = { &l->q, &l->k, &l->v, &l->o, &l->sh_g, &l->sh_u, &l->sh_d,
                                 &l->dn_qkv, &l->dn_z, &l->dn_out };
            for (int qw = 0; qw < 10; qw++)
                if (cq[qw]->map) madvise((void*)(((uintptr_t)cq[qw]->map) & ~(uintptr_t)4095),
                                         (size_t)cq[qw]->map_len + 4096, MADV_DONTNEED);
        }
#endif
        if (allow_prefetch && g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_trunk_stream: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".prestream"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_trunk_stream applied (%d edits, backup %s)" % (len(EDITS), bak))
