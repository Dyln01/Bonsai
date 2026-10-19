#!/usr/bin/env python3
"""patch_mapsc.py — E8: restore per-row .qs scales on the TRUNK_MAP int4 mmap path.

Bug (found 2026-10-19 while gating the prefix-reuse serve path): load_tq's
mmap branch reads the packed row's f32 scale vector into out->sc and then the
next line clobbers it (`out->w = NULL; out->q = NULL; out->sc = NULL;`), so
matmul_map_i4 ran with rs = 1.0 for every row — per-row amax/7 scales silently
dropped from EVERY dense int4 TRUNK_MAP run (all LOWRAM/BENCH dense numbers
pre-fix; relative A/B arms within one container stay comparable, absolute
quality was distorted, and hot-neuron energies were ranked on distorted
activations).

Fix: keep out->sc for i4 mapped rows; only the f16-map and non-map paths null
it. Gate (CI): a scales-=1.0 copy of the student container must now produce
DIFFERENT logits from the real-scaled one (pre-fix they were identical because
scales were ignored); plus the standing identity gates.

Anchored exact-byte edit, house style. Backup qwen36.c.premapsc.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "if (!is_i4) out->sc = NULL;" in src:
    raise SystemExit("patch_mapsc: already applied")

OLD = """                    out->w = NULL; out->q = NULL; out->sc = NULL;
                    out->I = I; out->O = O;
"""
NEW = """                    if (!is_i4) out->sc = NULL;   /* E8: i4 mapped rows KEEP .qs scales */
                    out->w = NULL; out->q = NULL;
                    out->I = I; out->O = O;
"""
if src.count(OLD) != 1:
    raise SystemExit("patch_mapsc: anchor count=%d (want 1)" % src.count(OLD))
src = src.replace(OLD, NEW, 1)

bak = P + ".premapsc"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mapsc applied (1 edit, backup %s)" % bak)
