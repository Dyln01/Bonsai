#!/usr/bin/env python3
"""patch_embed_map.py — Q36_EMBED_MAP=1 (Q36_EMBED_STREAM is an accepted alias): keep the embedding table on disk, read rows on demand.

WHY. `model.embed_tokens.weight` is vocab*hidden = 1.27e9 elements at the 27B geometry:
2.54 GB as f16, 5.09 GB as f32. The engine loads the whole table resident
(`load_t_n` -> falloc + st_read_f32) to serve ONE 20 KB row per decode token, and on a
7.62 GB mini-PC that is a third of RAM doing nothing. `project_q36.py --floor` prices the
freed RAM: at the field geometry, streaming the embed row adds 2.54 GB to `cache_room`,
which takes the streamed set from 4.93 GB/token to 2.38 GB (alone), and to ZERO with an
int4 lm_head -- i.e. off the disk-bound floor entirely.

HOW. `st_read_slice_f32(S, name, elem_off, n_elems, out, drop)` already exists in st.h:
bounds-checked, f16/bf16->f32, over the tensor's own persistent fd (`t->fd`,
`st_pread_full`), so a row read is one pread of D*esz bytes with no file re-open. The
table is touched in exactly three places (verified by enumerating every `m->embed`/
`m->embed_h` use), and this patch routes all three:

  1. the load site          -> skip the resident alloc, remember the resolved tensor name
  2. q36_embed_row()        -> add the mapped branch (ONE place knows how the table is stored)
  3. the prefill row loop   -> call q36_embed_row() instead of inlining the two cases,
                               so mapped mode works in prefill too
  plus the `dense_bits()==16` f16-conversion block, which must not run on a NULL table.

Refusals, not silent fallbacks: a table that is not a float tensor (int4/affine) cannot be
sliced and falls back to the resident load with a message; a missing tensor exits.

Byte-exact anchored edits, house style; backup qwen36.c.preebmap. Line endings are
normalised to LF before anchoring (a CRLF working tree silently no-ops multi-line anchors
while single-line ones match, which produced a half-applied patch once already).
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
raw = open(P, "rb").read()
src = raw.decode("utf8", "surrogateescape").replace("\r\n", "\n").replace("\r", "\n")
crlf = raw.count(b"\r\n")
if "g_q36_embed_map" in src:
    raise SystemExit("patch_embed_map: already applied")

EDITS = []


def edit(tag, old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        raise SystemExit(f"patch_embed_map: anchor '{tag}' matched {n} time(s), expected {count}")
    src = src.replace(old, new, count)
    EDITS.append(tag)


# 1) the globals + knob + the evidence line, at top level before the load site.
#    Anchored on load_t_n's own definition: top level, before every use site.
edit("globals",
     "static float *load_t_n(Model *m, const char *name, int64_t want) {",
     """/* ---- Q36_EMBED_MAP: the embedding table stays on disk, rows read on demand ----
 * vocab*hidden at the 27B geometry is 2.54 GB f16 / 5.09 GB f32 held resident to serve
 * one 20 KB row per decode token. Freed, the page cache holds that much more of the
 * per-token working set (see project_q36.py --floor: +2.54 GB of cache_room). */
static int  g_q36_embed_map = -1;                 /* -1 unread, 0 off, 1 on */
static char g_embed_name[QW_DENSE_NAME_MAX];      /* the resolved tensor name */
static long long g_embed_rows = 0, g_embed_bytes = 0;

static int q36_embed_map_on(void) {
    if (g_q36_embed_map < 0) {
        /* Q36_EMBED_STREAM is accepted as an alias: an independent sketch for this same
         * lever shipped under that name, so both spellings work and neither surprises. */
        const char *e = getenv("Q36_EMBED_MAP");
        if (!e || !*e) e = getenv("Q36_EMBED_STREAM");
        g_q36_embed_map = (e && e[0] == '1');
        if (g_q36_embed_map) atexit(q36_embed_report);   /* defined below; declared here */
    }
    return g_q36_embed_map;
}

static float *load_t_n(Model *m, const char *name, int64_t want) {""")

# 2) the load site: mapped mode skips the resident alloc.
edit("load site",
     '        m->embed = load_t_n(m, "model.embed_tokens.weight", (int64_t)c->vocab * c->hidden);',
     """        if (q36_embed_map_on()) {
            const char *en = dense_resolve(m, "model.embed_tokens.weight",
                                           g_embed_name, sizeof g_embed_name);
            /* dense_resolve may return ITS OWN buffer rather than the one passed in (it does
             * for the common name), so copy explicitly -- otherwise g_embed_name stays empty,
             * the mapped branch is skipped, and the forward memcpys from a NULL table.
             * ASan caught exactly that: q36_embed_row -> memcpy -> SEGV on +0x2a000. */
            snprintf(g_embed_name, sizeof g_embed_name, "%s", en);
            st_tensor *et = st_find(&m->S, en);
            if (!et) { fprintf(stderr, "missing %s\\n", en); exit(1); }
            if (et->dtype >= 3) {   /* int4/affine tables cannot be sliced as floats */
                fprintf(stderr, "[embed] Q36_EMBED_MAP: %s is %s, not a float tensor -- "
                                "keeping it resident\\n", en, st_dtype_name(et->dtype));
                m->embed = load_t_n(m, "model.embed_tokens.weight", (int64_t)c->vocab * c->hidden);
            } else {
                m->embed = NULL; m->embed_h = NULL;
                fprintf(stderr, "[embed] Q36_EMBED_MAP: %s stays on disk (%.2f GB not "
                                "resident); rows read on demand\\n",
                        en, (double)et->numel * st_dtype_esz(et->dtype) / 1e9);
            }
        } else {
            m->embed = load_t_n(m, "model.embed_tokens.weight", (int64_t)c->vocab * c->hidden);
        }""")

# 3) the f16-conversion block must not dereference a mapped (NULL) table.
edit("dense_bits16 guard",
     "        if (quantize_dense && dense_bits() == 16) {   /* 16-bit trunk: the table too, 2.5 GB less on the 27B */",
     "        if (quantize_dense && dense_bits() == 16 && m->embed) {   /* 16-bit trunk: the table too, 2.5 GB less on the 27B; NULL under Q36_EMBED_MAP */")

# 4) the read helper: one place knows how the table is stored. Also harden the vis_map
#    index against a negative pos (the prefill call site passes -1 to mean "no position").
edit("helper",
     """static void q36_embed_row(Model *m, int id, int pos, float *row) {
    int D = m->c.hidden;
    int vrow = (m->vis_map && pos < m->vis_map_len) ? m->vis_map[pos] : -1;
    if (vrow >= 0 && vrow < m->vis_rows_n)
        memcpy(row, m->vis_rows + (int64_t)vrow*D, D*sizeof(float));
    else if (m->embed_h)
        f16_to_f32_bulk(m->embed_h + (int64_t)id*D, row, D);
    else
        memcpy(row, m->embed + (int64_t)id*D, D*sizeof(float));
}""",
     """static void q36_embed_row(Model *m, int id, int pos, float *row) {
    int D = m->c.hidden;
    /* pos >= 0 guards the vis_map index: the prefill path passes -1 meaning "no position",
     * and vis_map[-1] was a latent out-of-bounds read. */
    int vrow = (m->vis_map && pos >= 0 && pos < m->vis_map_len) ? m->vis_map[pos] : -1;
    if (vrow >= 0 && vrow < m->vis_rows_n)
        memcpy(row, m->vis_rows + (int64_t)vrow*D, D*sizeof(float));
    else if (m->embed_h)
        f16_to_f32_bulk(m->embed_h + (int64_t)id*D, row, D);
    else if (!m->embed && g_embed_name[0]) {          /* Q36_EMBED_MAP: one row off disk */
        st_read_slice_f32(&m->S, g_embed_name, (int64_t)id*D, D, row, 0);
        long long nb = (int64_t)D * sizeof(float);
        g_embed_rows++; g_embed_bytes += nb;
    } else {
        if (!m->embed) {
            fprintf(stderr, "embed: no resident table and no mapped name -- refusing\\n");
            exit(1);
        }
        memcpy(row, m->embed + (int64_t)id*D, D*sizeof(float));
    }
}

/* atexit hook (registered by q36_embed_map_on): the evidence that the mapped path ran,
 * in the same spirit as the I/O shim's IOTH_STATS accounting -- an idle table and a
 * table that was never touched look identical without it. */
static void q36_embed_report(void) {
    if (g_embed_rows)
        fprintf(stderr, "[embed] mapped: %lld row read(s), %.1f MB off disk "
                        "(the table was never resident)\\n",
                g_embed_rows, (double)g_embed_bytes / 1e6);
}""")

# 5) the prefill row loop routes through the helper.
edit("prefill",
     """        if (m->embed_h) f16_to_f32_bulk(m->embed_h + (int64_t)ids[s]*D, x + (int64_t)s*D, D);
        else memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));""",
     """        /* through the helper, so prefill works in mapped mode too (pos -1 = no visual row,
         * which is the behaviour this loop had before) */
        q36_embed_row(m, ids[s], -1, x + (int64_t)s*D);""")

# the report function is used before its definition in q36_embed_map_on -> add a declaration
edit("fwd decl",
     "static int  g_q36_embed_map = -1;                 /* -1 unread, 0 off, 1 on */",
     "static void q36_embed_report(void);               /* atexit hook, defined with the read path */\n"
     "static int  g_q36_embed_map = -1;                 /* -1 unread, 0 off, 1 on */")

if os.path.exists(P):
    shutil.copyfile(P, P + ".preebmap")
out = src if crlf == 0 else src      # always emitted LF (see the header)
open(P, "wb").write(out.encode("utf8", "surrogateescape"))
print(f"patch_embed_map: {len(EDITS)} edit(s) applied to {P}"
      + (f" (normalised from CRLF, {crlf} line(s))" if crlf else "")
      + f"\n  backup: {P}.preebmap")
for e in EDITS:
    print(f"  - {e}")
