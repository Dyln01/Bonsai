# Absorbed-rotation decode for KV_TQ codec 1 (Phase 3b-rot) — qwen36.c

**Date:** 2026-10-19 · status: **design pinned, implementation in same commit**
**Scope:** codec 1 (rotated-int4). Decode first (`patch_tq_absorb.py`), then the
flash prefill (`patch_tq_absorb_prefill.py`, same knob): query rows rotate once
per (s0, kvh) block, K/V dequant to rotated-space levels per block instead of
FWHT per row per block, online-softmax accumulators stay in rotated space
(rescales are linear) and unrotate once per (row, head) at the cx copy.
Codec 0 (PolarQuant) keeps today's math — its angle-dependent rotation tree is
not a fixed linear map (see finding above).

## The finding Phase 3a left behind

3a (GQA-absorbed decode tiles) removed the *per-head redundancy*: each K/V tile
is dequantized once per (kv-head, row) and serves the whole head group. What
remains in the TQ decode is **real per-row work**: `coli_q4_dequant_row` ends
with `coli_tq_unrotate` — a sign-flip + FWHT pass (O(n log n) butterflies) paid
on **every K row and every V row of every decode step**. That, not redundancy,
is TQ4's residual ~4x decode gap vs KV8 (docs/KVTQ_QWEN36.md, post-3a numbers in
docs/GQA_ABSORB_QWEN36.md).

## Why it can be absorbed exactly

The rotation is a **fixed orthogonal map**: `coli_tq_rotate` = deterministic
sign flips (COLI_TQ_SEED) + normalized FWHT; `coli_tq_unrotate` is its inverse.
Codec 1 stores, per row, int4 Lloyd-Max levels of the *rotated* coordinates
plus the radius in the KV8 scale slot. So with `R = rotate`, `d̂ = levels*std`
(the rotated-space row, std = radius/√n):

    K_orig = R⁻¹ d̂_K        score = q·K_orig = (R q)·d̂_K
    V_orig = R⁻¹ d̂_V        Σ a_t V_orig,t = R⁻¹ (Σ a_t d̂_V,t)

i.e. **rotate each query head's row once per step**, dot against raw
level-dequantized K rows (no FWHT anywhere in the row loop), accumulate the
softmax-weighted V rows **in rotated space**, and **unrotate once per head per
step**. Per-row decode cost becomes KV8-class: nibble → codebook level →
multiply-accumulate. Lossless up to fp re-association (same products, different
sum order) — hence numeric-parity gates, not bit-identity gates.

PolarQuant (codec 0) reconstructs through an **angle-dependent** rotation tree
(M(angles)·e₁), which is not a fixed linear map: not absorbable. Prefill could
absorb the same way (online-softmax rescales are linear) but the flash tile
loop is not where TQ pays; left as a follow-up.

## Implementation contract (`tools/patch_tq_absorb.py`)

- Knob `Q36_TQ_ABSORB=0` restores the 3a loop verbatim (A/B + fallback);
  default on, active only when `kv_tq && tq_codec==1` and `q_per_kv*kvd ≤ 8192`
  (stack staging bound).
- New helper `q36_q4_rot_noscale(packed, dst, n)`: `dst[i] = coli_q4_lev[nib] / √n`
  — the radius-1 rotated-space row (radius stays hoisted in `Ksc/Vsc` exactly
  as today).
- In the 3a decode branch: rotate the group's query rows into a per-thread
  stack buffer before the K tile loop; K dequant + score dots use the rotated
  pair; V dequant stays rotated; after the V tile loop, unrotate each head's
  accumulator once, then the existing memcpy to `ctx`.
- Softmax, scale hoisting, t-ascending accumulation, tile blocking: untouched.

## Gates

1. **Numeric (C, no container):** `engine/tq_absorb_parity.c` in `gates.yml` —
   score and rotated-space V-accumulation match the reference dequant path to
   1e-3 rel for n=64/128. This is where numeric truth lives: random-weight
   synthetic containers are *mechanical* oracles only (a real-head student run
   explodes to 1e8 residuals by construction — lesson recorded in ROADMAP
   rules, 2026-10-19).
2. **Mechanical (CI engine-build):** student + `KV_TQ=4 KV_TQ_CODEC=1`,
   `Q36_TQ_ABSORB=0` vs `1`: `DUMP_IDS` sequences equal (both arms exercise the
   full TQ decode; the absorb arm additionally runs rotate/unrotate once per
   head-step).
3. `kv_tq_parity_q36` unchanged (codec math untouched).
3. Measured decode tok/s on real hardware: the KVTQ battery arms E/F/G with
   `Q36_TQ_ABSORB=0/1` — numbers land in the 2b-4 battery run (mini-PC),
   expectation: TQ4c1 decode moves from ~5x KV8's cost toward KV8-class at
   half KV8's bytes.
4. Bit-neutrality of the *off* path: all existing identity gates re-run.

## Follow-ups (Phase 3b remainder)

- **3b-scan:** the O(T) decode scan itself (57.65 ms/tok at the 68.4k fixture
  post-3a) — windowing/sparse KV (OasisKV-class, drafts predict rows) belongs
  here or in Phase 4; orthogonal to this patch.
- **3b-prefill: DONE (same commit series)** — flash tile loop absorbed as above;
  CI mechanical gate (absorb 0/1 id-identity with flash on) covers it; TTFT win
  measured on real hardware in the 2b-4 battery (expect the 1.4-1.6x TQ TTFT
  tax to shrink toward the KV8 class).
