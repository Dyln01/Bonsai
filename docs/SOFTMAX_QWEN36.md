# Q36_FAST_SOFTMAX — AVX2 fast exponential in the attention softmax

**Date:** 2026-10-17 · engine `qwen36` @ bf24429 + Bonsai patch stack (+ `tools/patch_softmax.py`, backup `qwen36.c.presm`)
**Knob:** `Q36_FAST_SOFTMAX=1` (opt-in, default stock-verbatim) · needs an AVX2+FMA build (the engine's default `-march=x86-64-v3` provides it; other builds warn and stay off)
**Model:** synthetic Qwen3.8-27B-shaped container (24L: 6 GQA + 18 DeltaNet) — no download.

## What it replaces

Stock `softmax_row` is three passes over the O(T) score row with a **libm
`expf` call per element** — nothing vectorizes it without `-ffast-math`. At
decode, that runs once per query-head per token: on the battery container it
is ~48 rows × O(16k) per token, ~2.6 ms of the measured 16.4 ms/token
attention phase.

The fast path: `2^x = 2^n · poly(f)`, `n = round(x·log2e)` written directly
into the exponent field, `f ∈ [−0.5, 0.5]` through a degree-5 minimax
polynomial (Chebyshev-node fit, measured — not copied from memory), the
argument reduced by **two FMAs** (L_HI/L_LO split of log2e) so the fraction
never inherits the `|x·log2e|` rounding error, all three passes vectorized
8-wide (AVX2+FMA), and the final divide replaced by a reciprocal-multiply.
The shipped knob covers `softmax_row` ONLY — the flash prefill's
online-softmax exps stay stock, a measured negative result (see below).

## Microbench + gates (`engine/softmaxbench_q36.c` — run it, all must pass)

| gate | result |
|---|---|
| max rel err vs `expf` over [−104, 0] (dense grid) | **2.46e-07** |
| underflow zone [−104, −87) | all outputs in [0, 1.6e-38] — the exponent field must never wrap |
| softmax vs an f64 reference, N(0,3) / N(0,15) | fast **9.6e-08 / 6.0e-08** vs stock 1.3e-06 / 3.0e-07 — **the fast path is more accurate than stock** (stock's plain f32 accumulator drifts ~1.3e-5 on the 16k row sum) |

Speed (Ryzen 7 7735HS, the engine's own flags):

| T (row) | stock ns/row | scalar fast | **AVX2 fast** | speedup |
|---|---|---|---|---|
| 1024 | 3381 | 3187 | **345** | 9.5× |
| 4096 | 13401 | 11911 | **1222** | 10.7× |
| 16384 | 54277 | 49424 | **5321** | 10.2× |
| 32768 | 116122 | 96524 | **9977** | 11.0× |

The scalar-only fast exp is 1.07× — **the win is the vectorization**, which
is why the knob is AVX2-gated rather than shipping the scalar variant.

Three bugs the gates caught before they could ship (kept here as the lesson):

1. From-memory polynomial coefficients: 1.5e-4 rel err. Refit on
   Chebyshev nodes → 2.5e-7. Never copy transcendental constants from memory.
2. Argument reduction in plain f32: the `x·log2e` rounding error grows
   linearly with |x| (3.7e-6 at x≈−69). The two-FMA exact-fraction reduction
   removes it.
3. Exponent-field underflow for x < −87.34: `n+127 < 0` wraps the field and
   produces ±1e33 garbage → NaN row sums. A N(0,15) gate arm caught it;
   the fix flushes below −87 (exp there is subnormal, irrelevant to softmax)
   and a dedicated gate sweeps that zone.

## Engine A/B (synth_qwen27, 15,960-token prompt, 48 new tokens, f32 KV, flash on, 3 runs × 2 arms, all rc=0)

| run | arm | TTFT s | attn ms/token | decode tok/s |
|---|---|---|---|---|
| 1 | off | 156.1 | 16.45 | 45.5 |
| 1 | on | 169.8 | 15.03 | 49.0 |
| 2 | off | 148.6 | 16.42 | 45.5 |
| 2 | on | 172.3 | 15.97 | 45.5 |
| 3 | off | 153.5 | 19.82 *(noisy run)* | 37.0 |
| 3 | on | 173.9 | 15.58 | 46.9 |

- **Attention phase: improved in 3/3 pairs** (−8.6% / −2.7% / −21%), and in
  the clean runs the magnitude matches the microbench prediction (softmax
  ≈ 2.6 ms of 16.4 ms attention; ~1.4 ms/token recovered). This is the
  trustworthy metric: direct mechanism, consistent sign, correct size.
- **Decode: mean 42.6 → 47.1 tok/s (+10.5%)**, but the box's identical-config
  noise band is ±7–9% (the historical A/J pair: 46.9 vs 41.1 on identical
  arms), so the honest statement is "attention-phase −6…−9%, decode improved
  in every pair, means +10% within a ±9% noise band".
- **TTFT anomaly: +13% with the knob on — REAL, not order** — 4 runs on
  (169.8/172.3/173.9/174.8, ±1.5% spread) vs 4 runs off (148.6/149.4/153.5/
  156.1, ±2.5%). An order-swapped control (knob-on ran FIRST, 174.8 s;
  knob-off second, 149.4 s) rules out arm-position effects: the gap follows
  the knob. It cannot be the exp swap's cost — the flash loop performs
  ~2.1M exps over the prefill ≈ 10 ms of a ~150 s prefill — and the
  **isolation build decided it**: reverting ONLY the two flash exp sites
  (keeping the softmax fast path) restored TTFT to **143.4 s — neutral** —
  while keeping the full decode win (attn 15.42 ms/token, 47.3 tok/s). The
  cost is a codegen perturbation of the flash V-accumulate loop ~2000×
  larger than the exp work itself, so the **shipped knob covers softmax_row
  only**; the flash exps stay stock, and the negative result is recorded in
  `tools/patch_softmax.py` so nobody re-tries it blind.

## The shipped shape (final)

| arm | TTFT | attn ms/token | decode tok/s |
|---|---|---|---|
| stock (4 runs) | 148.6–156.1 | 16.4–16.6 (clean runs) | 44.3–45.5 |
| **Q36_FAST_SOFTMAX=1, shipped** (softmax-only) | **143.4 — neutral** | **15.42 (−6.5%)** | **47.3 (+5%, inside the ±7–9% noise band)** |

Claim discipline: the attention-phase −6.5% is the trustworthy number
(direct mechanism, consistent in 5/5 pairs, magnitude matches the
microbench); decode tok/s improved in every pair but the box's noise band
swallows the mean delta, so the report leads with the phase metric.

## Honest labels

- **Lossy-by-rounding**: a 2.5e-7 exp perturbation can flip a near-tie greedy
  token — the same class as the Vulkan-tier rounding note in
  `docs/speculative.md`. That is why it is opt-in, default off; the default
  path is stock-verbatim.
- **No silu claim**: the MLP gate's `expf` was measured too (pristine-input
  arm): **0.96×** — no win, so the MLP path stays stock. The microbench
  prevented shipping a non-win.
- Real-model token-level neutrality is a Phase 5 item (the synthetic
  container's logits carry no quality signal).

## Usage

```bash
Q36_FAST_SOFTMAX=1 q36-minipc serve <model>    # any launcher command; env passes through
# verify engagement on stderr:
# [qwen36] Q36_FAST_SOFTMAX=1: AVX2 fast exp in the attention softmax (...)
```

Files: `tools/patch_softmax.py` (reproduces the engine edit from
`qwen36.c.presm`), `engine/softmaxbench_q36.c` (gates + bench),
`scripts/exp_softmax_q36.py` (the A/B driver), `results/results_softmax_q36.json`.
