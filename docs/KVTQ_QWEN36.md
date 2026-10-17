# KV_TQ — the third KV tier for qwen36 (PolarQuant + rotated-int4)

**Date:** 2026-10-16 · engine `qwen36` @ bf24429 + Bonsai patch stack (TRUNK_MAP/HOT/GRIFFIN/KV8/FLASH + KV_TQ)
**Codecs:** `kv_tq.h` (written for the GLM engine, wired into q36 here — both codecs ported)
**Model:** synthetic Qwen3.8-27B-shaped container (24L: 6 GQA + 18 DeltaNet, hidden 1024, kv_heads 4, head_dim 64) — no download.
**Host:** WSL2, Ryzen 7 7735HS (8C/16T), 16 GB RAM, 4-core profile.

## The tier ladder

| tier | row width (HD=64 / HD=128) | extra per row | codec error (mean rel L2) | loss |
|---|---|---|---|---|
| f32 (default) | 256 / 512 B | — | 0 | exact |
| KV8 (fp8-e4m3) | 64 / 128 B | f32 scale | ~0.02–0.06 (e4m3 half-step) | lossy, opt-in |
| **KV_TQ=4 codec 1** (rotated int4) | 32 / **64 B** | f32 radius | **0.096** (measured, parity gate) | lossy |
| KV_TQ=4 codec 0 (PolarQuant-4) | 28 / 56 B | f32 radius | 0.147 | lossy |
| KV_TQ=3 codec 0 (PolarQuant-3) | 20 / **40 B** | f32 radius | 0.291 | lossy |

How it works: a randomized Hadamard rotation makes row coordinates near-Gaussian,
then either a 16-level Lloyd-Max codebook scales the row by its std (codec 1) or
the recursive polar transform packs n−1 angles into 3–4 bits each (codec 0). The
row's L2 norm (radius) rides **the same per-row scale slot KV8 already stores**,
so every read path keeps KV8's scale-hoisting: decode the row at radius 1 (unit
direction) and multiply by `Kscale/Vscale` outside the element loop.

## Measured matrix (synthetic 27B-shaped, 15,960-token prompt, 48 new tokens, all rc=0)

All arms ran with the flash prefill ON (it is the engine default; `Q36_FLASH=0`
disables). Decode = honest `[timers] TOTAL ms/token` — the `Speed:` line includes
prefill and is never a decode number.

| arm | peak RSS | vs A | TTFT | decode tok/s | vs A |
|---|---|---|---|---|---|
| **A. f32 KV** (defaults) | 805.6 MB | — | 146.1 s | 46.9 | — |
| J. f32 KV (repeat) | 805.8 MB | 0 | 156.6 s | 41.1 | −12% (noise band) |
| **B. KV8** | 673.3 MB | **−132 MB (−16%)** | 132.4 s | 26.6 | −43% |
| **E. KV_TQ=4 codec 1** (rot-int4) | **649.9 MB** | **−156 MB (−19%)** | 163.1 s | 5.3 | −89% |
| F. KV_TQ=4 codec 0 (Polar-4) | 647.6 MB | −158 MB | 225.3 s | 1.2 | −97% |
| G. KV_TQ=3 codec 0 (Polar-3) | **641.3 MB** | **−164 MB (−20%)** | 229.6 s | 1.3 | −97% |
| **H. KV_TQ=4 c1 + int4 dense** | **573.6 MB** | **−232 MB (−29%)** | 157.4 s | 6.3 | −87% |

Reading of the deltas:

- **RSS ladder is exactly the codec ladder.** B→E→F→G shave 23.4 → 2.3 → 6.3 MB
  in lockstep with 64→32→28→20 B/row (HD=64). H vs the KV8+int4 profile from
  the LOWRAM matrix (595.6 MB): −22 MB with the same dense quant.
- **KV_TQ is a RAM lever, not a speed lever.** The per-row decode now includes
  an inverse Hadamard transform (n·log₂n butterflies) plus, for the polar
  codec, per-angle bit extraction and trig — paid on **every K row at S=1**
  where KV8's e4m3 decode is a table lookup. The result: rot-int4 decode is
  ~5× KV8's cost, polar ~20×. Same structural class as KV8's −43%: no MLA-style
  score absorption in this engine, and S=1 amortizes the whole dequant over
  only H/KVH query rows.
- **Flash+KV_TQ verified**: the flash tile kernel carries the packed-row stride
  and codec dispatch (arms E–I all flash-on, rc=0, RSS/decode unchanged vs
  non-flash peers). Flash also erases KV8's old prefill regression here
  (B TTFT 132 s vs f32's 146 s — the tile kernel amortizes dequant across the
  query block).
- **TTFT under polar codecs is 1.4–1.6× f32** (225–230 s): prefill dequants
  each row once per query block, and the polar decoder is trig-bound.

## The honest trade table

| Situation | Recommended tier | Why |
|---|---|---|
| RAM fits at your ctx | f32 | exact and fastest decode |
| RAM-bound, want the cheapest usable tier | **KV_TQ=4 codec 1** | −19% total RSS, −23 MB vs KV8 at 8k on the small container; decode 5.3 tok/s |
| Squeezing the last GB (eco / very long ctx) | KV_TQ=3 codec 0 | −20% total RSS, smallest rows that exist in the stack; interactive speed is gone (1.3 tok/s) |
| Combined floor | KV_TQ=4 c1 + COLI_DENSE_BITS=4 | 573.6 MB — 29% under defaults at this operating point |

**Scaled to real Qwen3.8-27B (64L, kv_heads 8, head_dim 128, 32k ctx —
extrapolation from measured ratios, same method as the LOWRAM report):** the
f32 KV pool is ~17.2 GB; KV8 cuts it to ~4.3 GB; **KV_TQ=4 codec 1 halves that
again to ~2.2 GB (+~0.27 GB radii)**, and Polar-3 lands ~1.4 GB. A 32k-context
session on an 8 GB box moves from "KV8 barely fits" to "KV_TQ fits with room
for the mapped trunk + hot set". Needs real-checkpoint confirmation like every
other row in this stack.

## Correctness gates (all green — `engine/kv_tq_parity_q36.c`)

- **FWHT**: self-inverse and norm-preserving at n=64/128.
- **Radius exactness**: `quant` returns ‖row‖ for both codecs; rides Kscale/Vscale.
- **Direction coverage**: rot-int4 0.096 < 0.15, polar-4 0.147 < 0.45, polar-3
  0.291 < 0.65 (thresholds shared with the GLM-side test; measured over 300
  Gaussian rows per width).
- **Scale-hoisting linearity** (the port's core premise): `dequant(r=1)·s ==
  dequant(r=s)` element-wise over 50 random scales per width per codec —
  bounded at 1e-5 rel, because the radius scales the reconstruction linearly.
- **Inert rows**: zero/NaN/Inf rows quantize to radius 0 and decode to exact
  zeros (poisoned rows never reach the FWHT).
- **Accounting**: polar-4 28/56 B < int4 32/64 B < fp8 64/128 B < f32 at both
  widths; int4 is exactly n/2.
- Engine-level: all 8 battery arms rc=0; default path byte-identical when
  `KV_TQ` unset (no code-path change without the env).

## Usage

```bash
KV_TQ=4 KV_TQ_CODEC=1   # rotated int4 (default codec) — the usable tier
KV_TQ=3 KV_TQ_CODEC=0   # PolarQuant-3 — the RAM extreme
# or via the launcher:
Q36_KVQ=tq4 q36-minipc serve ~/model    # f32 | kv8 | tq4 | tq3
```

Mutually exclusive with KV8; `KV_TQ` wins if both are set. The engine refuses
non-power-of-two head dims (codec would silently zero rows) and prints a
banner with the resolved row width — read it before trusting a run.

## Next steps

1. **The decode cost is the whole remaining story, and it is architectural,
   not codec**: an absorbed-score (FlashMLA-style) attention path would amortize
   the row dequant across query heads and, with speculative KV-row prediction
   (OasisKV-style), skip most of the O(T) scan — that is Phase 3 of the roadmap
   and now carries KV_TQ as well as KV8.
2. Codec-level speedups worth a day each: cache the unrotated staging per tile
   across query heads (K rows serve H/KVH=8 heads on the real 27B, vs 2 here),
   and a LUT for the polar angle decode.
3. Real-checkpoint perplexity at each tier (gated: needs a real download).
