#!/usr/bin/env python3
"""patch_calib_bar.py — E7: prefill progress lines + periodic profile saves.

The engine printed NOTHING during prefill (timers only at the end), which made
calibrate/bench on dense 27B look hung on small boxes. Adds, stderr-only:

  [prefill] p=<percent> layer=<done>/<total> t=<elapsed_s> eta=<eta_s>

one line per layer of every multi-row forward (S>1), plus a periodic
q36_hot_profile_save() every 8 layers when HOT_PROFILE is set, so a calibrate
run killed mid-pass still leaves a usable (partial-token) profile on disk.

Knob: Q36_PROGRESS=0 disables the lines. Compute is untouched — the CI
identity gates (DUMP_IDS / logits sha) prove bit-neutrality.

Anchored exact-byte edits, house style. Backup qwen36.c.prebar.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "g_q36_progress" in src:
    raise SystemExit("patch_calib_bar: already applied")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

edit("""static int  g_q36_mtp_on = 0;
""",
     """static int  g_q36_progress = 1;   /* Q36_PROGRESS=0 disables prefill progress lines */
static int  g_q36_mtp_on = 0;
""")

edit("""      if (mf && *mf) snprintf(g_q36_mtp_force, sizeof g_q36_mtp_force, "%s", mf); }
""",
     """      if (mf && *mf) snprintf(g_q36_mtp_force, sizeof g_q36_mtp_force, "%s", mf); }
    { const char *pv = getenv("Q36_PROGRESS"); if (pv && pv[0] == '0') g_q36_progress = 0; }
""")

edit("""    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = layer_begin; i < layer_end; i++) {
""",
     """    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    double _pt0 = (S > 1 && g_q36_progress) ? now_s() : 0.0;
    for (int i = layer_begin; i < layer_end; i++) {
""")

edit("""        if (allow_prefetch && g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
""",
     """        if (allow_prefetch && g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
        if (S > 1 && g_q36_progress) {   /* E7: the bar feed — stderr only, no compute */
            int done = i + 1 - layer_begin, tot = layer_end - layer_begin;
            double el = now_s() - _pt0;
            fprintf(stderr, "[prefill] p=%d layer=%d/%d t=%.0f eta=%.0f\\n",
                    done * 100 / tot, done, tot, el, el * (tot - done) / (done > 0 ? done : 1));
            fflush(stderr);
        }
        if (S > 1 && g_hot_profile_path[0] && ((i + 1) & 7) == 0)
            q36_hot_profile_save();      /* a killed calibrate keeps a partial profile */
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_calib_bar: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".prebar"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_calib_bar applied (%d edits, backup %s)" % (len(EDITS), bak))
