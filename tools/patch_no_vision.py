#!/usr/bin/env python3
"""patch_no_vision.py — Q36_NO_VISION=1: do not read the vision tower into RAM.

Field log (bench_20261116-150228): the engine's anonymous memory peaks at ~5.0 GB right after
load on an 8 GB box, against the ~0.7 GB the cost model assumed. model-globals.safetensors
(3.18 GB: the embedding table et al.) and model-vision.safetensors (0.92 GB) are both read into
private RAM. The vision tower is never used by a text-only run, yet q36_load_vision() pulls all
27 blocks in unconditionally. That RAM is exactly what the mapped trunk's page cache needs.

The engine already has a text-only path for a container with no tower (vis_depth = 0 and an
early return; m->vis_ready stays 0, so an image request is refused with a clean ERROR). This
patch takes the same path on request. OFF by default: serving with images is unaffected.

One anchored edit, house style. Backup qwen36.c.prenovis.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "Q36_NO_VISION" in src:
    raise SystemExit("patch_no_vision: already applied")

OLD = """static void q36_load_vision(Model *m) {
    Cfg *c = &m->c;
    if (!c->vis_depth) return;
"""
NEW = """static void q36_load_vision(Model *m) {
    Cfg *c = &m->c;
    if (!c->vis_depth) return;
    {   /* Q36_NO_VISION=1: text-only run — do not read the ~0.9 GB tower into private RAM. Same
         * state as a container without one: vis_ready stays 0 and images are refused cleanly. */
        const char *nv = getenv("Q36_NO_VISION");
        if (nv && nv[0] == '1') {
            fprintf(stderr, "[qwen36] vision tower skipped (Q36_NO_VISION=1): text only, images refused\\n");
            c->vis_depth = 0; return;
        }
    }
"""
n = src.count(OLD)
if n != 1:
    raise SystemExit(f"patch_no_vision: q36_load_vision anchor found {n}x (want 1) — wrong qwen36.c?")
open(P, "wb").write(src.replace(OLD, NEW, 1).encode("utf8", "surrogateescape"))
print("patch_no_vision: applied (Q36_NO_VISION=1 skips the vision tower)")
