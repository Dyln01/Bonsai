#!/usr/bin/env python3
"""patch_ftz.py — flush-to-zero / denormals-are-zero on every FP thread.

Subnormal floats trap into microcode (~50-100x per op). One mis-scaled weight
slab in a real converted shard, fed into a long recurrence chain (DeltaNet),
can turn a single layer into an hours-long crawl while every gate on synthetic
containers stays green (well-scaled draws never produce subnormals). FTZ/DAZ
makes |x| < ~1e-35 flush to zero — noise by construction for inference — and
restores constant-time FP.

MXCSR is per-thread: set on main at startup and once on the whole OMP pool
(the pool size is fixed for the run — no num_threads clauses anywhere — so
every later worker inherits it). CI identity gates prove bit-no-change on
clean data.

Anchored exact-byte edit, house style. Backup qwen36.c.preftz.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "q36_ftz_daz" in src:
    raise SystemExit("patch_ftz: already applied")

OLD = "int main(int argc, char **argv) {\n"
NEW = """/* Flush-to-zero + denormals-are-zero for this thread's MXCSR. See
 * patch_ftz.py docstring: subnormals in a recurrence chain are an hours-long
 * crawl, not a numerics feature. */
static void q36_ftz_daz(void) {
#if defined(__SSE__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

int main(int argc, char **argv) {
    q36_ftz_daz();
#ifdef _OPENMP
    #pragma omp parallel
    { q36_ftz_daz(); }   /* whole pool, once; pool size is fixed for the run */
#endif
"""
if src.count(OLD) != 1:
    raise SystemExit("patch_ftz: anchor count=%d (want 1)" % src.count(OLD))
src = src.replace(OLD, NEW, 1)

bak = P + ".preftz"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_ftz applied (1 edit, backup %s)" % bak)
