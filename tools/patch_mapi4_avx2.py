#!/usr/bin/env python3
"""patch_mapi4_avx2.py — vectorize the TRUNK_MAP int4 matmul (matmul_map_i4).

Field logs (hot_check_1..5) show the mapped trunk spending its time in page
faults; the cost model had no compute term for what comes after. The compute
side is not free: matmul_map_i4 stages each output row through a per-nibble
scalar loop (branchy sign-extend, int->float, then a separate scale pass).
Measured in isolation that is ~1.07 ns/weight (~2.2 cycles/weight) = ~3.6 s
per forward on 4 cores at the 27B operating point (~13.7 G weights touched),
even with every weight already resident. It becomes the next wall the moment
the working set fits in RAM.

Three edits, all in matmul_map_i4's neighbourhood:

  1. #include "mapi4_avx2.h" (new header, copied by the engine-build glob).
  2. TIER 1 — the staging loop becomes q36_m4_stage_row(): AVX2 with run-time
     dispatch (function-level target attributes, so it works whatever -march
     the engine was built with), producing the SAME floats as the scalar
     loop, hence bit-identical outputs through the unchanged dot_f32_lanes.
     Q36_MAPI4_SCALAR=1 restores the original loop.
  3. A one-row call must not pay for a thread team: `omp parallel` becomes
     `omp parallel if (O >= 32)`. The GRIFFIN gather calls matmul_d once per
     neuron row (O = 1; ~560k calls per token at frac 0.25), each of which
     used to spawn a team and malloc a staging buffer. Per-row results are
     unchanged (each row is always computed by exactly one thread, same order).
  OPT-IN — Q36_MAPI4_IDOT=1 routes I % 64 == 0 matrices through an int8
     activation / integer-SIMD kernel (NOT bit-exact; validate perplexity).

Gate: engine/mapi4_parity.c (bit-exact staging, bounded idot error, batch and
thread independence). Anchored exact-byte edits, house style.
Backup qwen36.c.premapi4.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "mapi4_avx2.h" in src:
    raise SystemExit("patch_mapi4_avx2: already applied")

# ---- 1. include ----
OLD_INC = '#include "kv8_fused.h"\n'
NEW_INC = ('#include "kv8_fused.h"\n'
           '#include "mapi4_avx2.h"  /* AVX2 staging / integer dot for matmul_map_i4 (Q36_MAPI4_SCALAR=1, Q36_MAPI4_IDOT=1) */\n')

# ---- 2+3. the kernel ----
OLD_FN = r"""static void matmul_map_i4(float *y, const float *x, const QW *w, int S, int I, int O) {
    const uint8_t *W = (const uint8_t *)w->map;
    const float *sc = w->sc;
    #pragma omp parallel
    {
        float *st = malloc((size_t)I * sizeof(float));
        if (!st) { fprintf(stderr, "OOM map stage\n"); exit(1); }
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *row = W + (int64_t)o * (I >> 1);
            for (int i = 0; i < I; i += 2) {
                uint8_t b = row[i >> 1];
                int8_t e0 = (int8_t)(b & 0xF);  if (e0 & 8)  e0 -= 16;
                int8_t e1 = (int8_t)(b >> 4);   if (e1 & 8)  e1 -= 16;
                st[i] = (float)e0; st[i+1] = (float)e1;
            }
            float rs = sc ? sc[o] : 1.0f;
            if (rs != 1.0f) for (int i = 0; i < I; i++) st[i] *= rs;
            for (int s = 0; s < S; s++)
                y[(int64_t)s * O + o] = dot_f32_lanes(x + (int64_t)s * I, st, I);
        }
        free(st);
    }
}
"""

NEW_FN = r"""static void matmul_map_i4(float *y, const float *x, const QW *w, int S, int I, int O) {
    const uint8_t *W = (const uint8_t *)w->map;
    const float *sc = w->sc;
    /* Q36_MAPI4_IDOT=1: int8-activation integer-dot kernel (opt-in, NOT bit-exact). Also
     * resolves the kernel mode once, before any thread team exists. */
    if (q36_m4_try_idot(y, x, W, sc, S, I, O)) return;
    /* mapi4 E26: a one-row call (the GRIFFIN gather issues one per neuron) must not spawn a
     * thread team and malloc per call; rows are still computed one-per-thread, same order. */
    #pragma omp parallel if (O >= 32)
    {
        float *st = malloc((size_t)I * sizeof(float));
        if (!st) { fprintf(stderr, "OOM map stage\n"); exit(1); }
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *row = W + (int64_t)o * (I >> 1);
            float rs = sc ? sc[o] : 1.0f;
            /* AVX2 staging, bit-identical to the scalar unpack + scale loop it replaced
             * (Q36_MAPI4_SCALAR=1 runs that loop). */
            q36_m4_stage_row(st, row, I, rs);
            for (int s = 0; s < S; s++)
                y[(int64_t)s * O + o] = dot_f32_lanes(x + (int64_t)s * I, st, I);
        }
        free(st);
    }
}
"""

for name, old in (("include anchor", OLD_INC), ("matmul_map_i4", OLD_FN)):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"patch_mapi4_avx2: {name} anchor found {n}x (want 1) — wrong qwen36.c?")

src = src.replace(OLD_INC, NEW_INC, 1).replace(OLD_FN, NEW_FN, 1)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mapi4_avx2: applied (matmul_map_i4: AVX2 staging + if(O>=32) team + opt-in idot)")
