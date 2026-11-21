#!/usr/bin/env python3
"""patch_down_stage.py — AVX2 staging for the hot-neuron down_proj (q36_down_prefix_batch).

Profile of a resident decode run (profile_20261121, gdb, 436 thread samples): q36_down_prefix_batch is the
single largest named hotspot — 14 % of all thread samples innermost, 25 % inclusive — although down_proj is
only ~8 % of the weights read per token. Its mapped-int4 branch still stages each row with the ORIGINAL
scalar loop (per-element `(i & 1)` branch, sign extend, int->float, then a separate scale pass): the same
loop E26 replaced in matmul_map_i4 and measured at ~1.07 ns/weight against ~0.1 ns/weight vectorized. It is
also the only down routine the decode path calls (one call site, S = rows verified, S >= 1).

The replacement is q36_m4_stage_prefix (mapi4_avx2.h): AVX2 with run-time dispatch, odd-Ik safe, producing
the SAME floats as the loop it replaces, so every output is bit-identical (the dot is unchanged). Works with
Q36_DOWN_COMPACT on or off; Q36_MAPI4_SCALAR=1 restores the scalar staging. No numerics decision involved.

One anchored edit, house style. Backup qwen36.c.predstage.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "q36_m4_stage_prefix" in src:
    raise SystemExit("patch_down_stage: already applied")

OLD = """                for (int i = 0; i < Ik; i++) {
                    uint8_t b = row[i >> 1];
                    int8_t e = (int8_t)((i & 1) ? (b >> 4) : (b & 0xF));
                    if (e & 8) e -= 16;
                    st[i] = (float)e;
                }
                float rs = sc ? sc[d] : 1.0f;
                if (rs != 1.0f) for (int i = 0; i < Ik; i++) st[i] *= rs;
"""
NEW = """                /* E29: AVX2 staging, bit-identical to the scalar unpack + scale it replaces (odd Ik safe) */
                q36_m4_stage_prefix(st, row, Ik, sc ? sc[d] : 1.0f);
"""
a = src.index("static void q36_down_prefix_batch(")
b = src.index("static void q36_down_prefix(", a)
chunk = src[a:b]
n = chunk.count(OLD)
if n != 1:
    raise SystemExit(f"patch_down_stage: scalar staging block found {n}x in q36_down_prefix_batch (want 1) — wrong qwen36.c?")
open(P, "wb").write((src[:a] + chunk.replace(OLD, NEW, 1) + src[b:]).encode("utf8", "surrogateescape"))
print("patch_down_stage: applied (q36_down_prefix_batch stages through q36_m4_stage_prefix)")
