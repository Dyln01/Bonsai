#!/usr/bin/env python3
"""patch_progress_dt.py — per-layer dt + kind on the [prefill] progress lines.

Field forensics for stall reports: "stuck at 87%" becomes "layer 56/64 kind=attn
dt=7200s", which separates a thrash (dt huge, disk busy), a bad-sector read
(dt huge, disk retrying one slab), a denormal crawl (dt huge, CPU busy — pre-FTZ)
and normal per-layer cost (dt ~ tens of s at calibrate token counts).

Anchored exact-byte edit, house style. Backup qwen36.c.preprog2.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "kind=%s dt=" in src:
    raise SystemExit("patch_progress_dt: already applied")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

edit("""    double _pt0 = (S > 1 && g_q36_progress) ? now_s() : 0.0;
""",
     """    double _pt0 = (S > 1 && g_q36_progress) ? now_s() : 0.0;
    double _lt0 = _pt0;
""")

edit("""        if (S > 1 && g_q36_progress) {   /* E7: the bar feed — stderr only, no compute */
            int done = i + 1 - layer_begin, tot = layer_end - layer_begin;
            double el = now_s() - _pt0;
            fprintf(stderr, "[prefill] p=%d layer=%d/%d t=%.0f eta=%.0f\\n",
                    done * 100 / tot, done, tot, el, el * (tot - done) / (done > 0 ? done : 1));
            fflush(stderr);
        }
""",
     """        if (S > 1 && g_q36_progress) {   /* E7: the bar feed — stderr only, no compute */
            int done = i + 1 - layer_begin, tot = layer_end - layer_begin;
            double el = now_s() - _pt0, dt = now_s() - _lt0;
            _lt0 = now_s();
            fprintf(stderr, "[prefill] p=%d layer=%d/%d t=%.0f eta=%.0f kind=%s dt=%.0f\\n",
                    done * 100 / tot, done, tot, el, el * (tot - done) / (done > 0 ? done : 1),
                    c->is_attn[i] ? "attn" : "dn", dt);
            fflush(stderr);
        }
""")

for old, new_ in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_progress_dt: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new_, 1)

bak = P + ".preprog2"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_progress_dt applied (%d edits, backup %s)" % (len(EDITS), bak))
