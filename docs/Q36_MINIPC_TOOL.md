# Small MoE models that slice like Qwen3.8-27B + the q36-minipc one-knob launcher

**Date:** 2026-10-14 · all numbers measured on this box (4 cores / 7.8 GB WSL emulation, D-disk).

## 1. Which "small total / tiny active" MoE models exist — and which colibri can slice TODAY

colibri's slicing works on any MoE whose router it speaks. Engines in-tree: `qwen36` (Qwen3 family),
`olmoe`, `glm53`, `inkling`, `kimi_k3`, `deepseek_v4/v41`, `laya`, `mimo`.

| Model | Total / active | Engine | Slices? | Fit on 8 GB mini PC |
|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B** | 30B / 3B | qwen36 | yes (measured: 8.2 tok/s @ cap 48) | **best coder-class pick** |
| **Qwen3.6-35B-A3B** | 35B / 3B | qwen36 | yes (the engine's reference MoE) | yes (~7.2 GB, same profile) |
| Qwen3-Coder-REAP-25B-A3B | 25B / 3B (103 experts) | qwen36 | yes | even lighter, coding-tuned |
| OLMoE-1B-7B | 7B / 1B | olmoe | yes | **the lightest**: 1.5-1.8 GB RSS, 4-19 tok/s measured upstream |
| Nemotron 3 Nano 30B-A3B | 30B / ~3B | — (hybrid Mamba) | needs a converter | reference only |
| Gemma 4 26B-A4B | 26B / 4B | — | no engine yet | — |
| Liquid LFM2-24B-A2B | 24B / 2.3B | — | no engine yet | — |

**Recommended ladder for mini PCs:** OLMoE-1B-7B (tiny/always-on) -> Qwen3-Coder-REAP-25B or 30B-A3B
(full coding agent). The two Qwen MoEs run through the same `q36-minipc` launcher unchanged.

## 2. q36-minipc: the web-ops techniques, mapped to engine mechanisms, in ONE knob

| Web technique | Engine mechanism (all pre-existing or patched this session) |
|---|---|
| CDN edge caching | expert LRU per layer (`--cap`) + `PIN=auto` + `.coli_usage` hot-set pinning |
| Anticipatory prefetch / preload | `PILOT=2` router-lookahead disk prefetch (71.6% one-layer-ahead predictability) |
| Adaptive quality (AV1 ladder) | `WIDE` adaptive expert-candidate window + `SMOOTH/CONF_LIMIT` confidence gate |
| Lazy loading | on-demand expert streaming (pread, default) |
| Compression (brotli/zstd) | int4-gs64 experts on disk + `KV8` KV cache (+ this session's KV8-native flash prefill) |
| Cache-busting guard | `CAP_RAISE=0` — never auto-grow past the measured RAM |
| Request coalescing / backpressure | `OMP_NUM_THREADS` cap (decode is memory-bound; threads beyond the RAM channels only add CPU heat) |
| TTL eviction | the per-layer LRU itself |

Install (already done in WSL):
```bash
cp q36-minipc.sh ~/bin/q36-minipc && chmod +x ~/bin/q36-minipc
q36-minipc seed  ~/qwen3-coder-i4        # once: Zipf .coli_usage prior (or skip; history also self-warms)
Q36_LEVEL=balanced q36-minipc bench ~/qwen3-coder-i4   # honest battery
Q36_LEVEL=balanced q36-minipc serve ~/qwen3-coder-i4   # long-running OpenAI-style serve
```

Levels (env `Q36_LEVEL`, or `Q36_RAM_GB` to override the auto RAM detect):
| level | budget | cap | threads | KV | ctx | measured (30B synthetic) |
|---|---|---|---|---|---|---|
| eco | ~5 GB | 24 | 2 | KV8 | 4k | 4.83 GB, 96.8% hit, ~1.4 tok/s decode, TTFT 20 s |
| balanced | ~7.2 GB | 48 | 4 | KV8 | 8k | 7.22 GB, 98.2-99.6% hit, 3.4-8.2 tok/s decode, TTFT 17 s |
| quality | needs ~12 GB | 64 | 4 | f32 | 16k | for bigger boxes |

Decode varies with cache warmth and workload locality (8.2 tok/s steady on Zipf chat-like traffic;
3.4 tok/s on the bench's cold random-ish code prompt). TTFT 17-20 s at ~350-token prompts with
PILOT=2 (vs 220 s cold without prefetch in the first battery — the prefetch + flash prefill are doing
their job).

## 3. What this adds over stock colibri

- One knob instead of 12 env vars, with a level table derived from the measured 8 GB ceiling
  (cap=64 thrashes at 7.3 GB — never offered; the launcher auto-caps on <8 GB boxes).
- `bench` prints the honest decode-phase tok/s (the `Speed:` line still conflates prefill).
- `seed` writes a calibrated Zipf prior so `PIN=auto` has data from token one on a fresh install.
- Composition verified: PILOT=2 costs nothing when warm (298 vs 313 ms/tok) and is what cut cold
  TTFT 13x on this box.
