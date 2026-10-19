# colibri RAM & Throughput Optimization Research
### Deep-dive with a synthetic testing harness — no real model weights used

> **Errata 2 (2026-10-19, E8):** dense int4 runs with `TRUNK_MAP=1` dropped the per-row
> `.qs` scales in the mapped GEMV until `patch_mapsc.py`; every dense number above is a
> relative A/B on scale-less weights (internally consistent, absolute quality superseded).
> Re-measurement rides along with the 2b-4 battery.
>
> **Errata (2026-10-19):** this report predates the Bonsai rebrand and the repo
> re-layout. Path map: `~/colibri-research/synthgen/make_synthetic_model.py` and
> `harness/*` → generators now live in `scripts/` (`gen_*.py|sh`,
> `run_workloads.py`, `make_tokenizer.py`); `harness/results_*.json` →
> `results/` (see `results/README.md`); `~/synth_models/` stays a local dir.
> `harness/iothrottle.c` was never committed — tracked as ROADMAP item E2.
> Numbers below are unaffected (they cite the JSONs, which moved with them).

**Date:** 2026-10-11 · **Engine audited/built:** JustVugg/colibri @ `bf24429` (v2.0.0, 2026-10-06), built from source in WSL2 Ubuntu (gcc 15.2, OpenMP), 700 KB binary
**Host:** AMD Ryzen 7 7735HS (8C/16T), 16 GB RAM, ext4 on Kingston SNV3S1000G NVMe (WSL2)
**Hard constraint honored:** no real checkpoints were downloaded. All experiments run against synthetic containers produced by a generator that emits colibri-native int4-gs128 safetensors with a Monte-Carlo-calibrated Zipf router. Every number below is either (a) measured on this harness, or (b) explicitly labeled extrapolation with its assumptions.

---

## 0. What was actually built (the synthetic environment)

| Component | File | What it does |
|---|---|---|
| Synthetic model generator | `~/colibri-research/synthgen/make_synthetic_model.py` (322 L, numpy-only) | Emits a colibri-loadable container: exact `config.json` keys the C loader reads, hand-written safetensors (no torch), fmt=4 int4-gs128 trunk/shared/experts, fmt=1 int8 embed/head, one shard file per sparse layer so experts genuinely stream from disk. Router bias calibrated by Monte-Carlo so the **top 25% of experts take ~81% of traffic** (empirical MoE shape). |
| Models on disk | `~/synth_models/` | `synth_std` (10L × 128 experts, 1.9 GB), `synth_fuge_trunk`→`synth_fat_trunk` (12L × 128E, hidden 4096, 4.6 GB), `synth_huge_trunk` (16L × 16E, hidden 8192 → 577 MB trunk, 2.1 GB) |
| I/O fidelity layer | `harness/iothrottle.c` → `iothrottle.so` | `LD_PRELOAD` shim interposing `pread/pread64/read`: per-read latency floor + bandwidth cap on files under the model dir, so the small synthetic container reproduces real-disk I/O. **Verified**: requested 50 MB/s → measured 49.4 MB/s, accounting exact. Emulation params: `IOTH_LAT_US=100`, `IOTH_BW_MBPS=3500` (PCIe4 NVMe class) / `500,150` (SATA class). |
| Workload driver | `harness/run_workloads.py` | Cold / steady / long-context profiles over REPLAY (fixed 512-token stream), records tok/s, hit %, GB read, p50/p90/p99/max latency, peak RSS (`/usr/bin/time -v`), pin/LRU residency |
| Focused experiments | `exp_kv_serve.py`, `exp_kv_and_trunk.py`, `exp_pin_recovery.py`, `compare_kv.py`, `verify_structural.py` | KV-pool RAM at CTX=131072 via SERVE mode; dense-trunk offload A/B; learned-pin recovery; KV-quantization output comparison; greedy determinism gate |
| Tokenizer | `harness/make_tokenizer.py` | Minimal byte-level `tokenizer.json` (256 byte tokens + EOS) enabling PROMPT/SERVE modes |

**Known harness limitations (documented, not hidden):**
- Page cache can't be dropped without root; instead `DIRECT=1` (O_DIRECT) is used so the throttle layer's latency/bandwidth actually apply. The `DIRECT=0` arm measures what a warm page cache does (it serves the whole 2 GB container at RAM speed — 119.9 tok/s).
- `io_uring` submits raw SQEs that bypass the `LD_PRELOAD` shim — the `URING=1` arm is effectively unthrottled raw-disk (still valid as an A/B against `PIPE=1` on real I/O, invalid as a throttled datapoint).
- Peak RSS on a 16 GB host under-counts the KV pool (lazy pages); the engine's own `[KV] projected pool` line is the honest metric for pool size.

**Harness verification gates (structural correctness, no perplexity needed):**
- **Determinism: PASS** — two fresh engine processes, greedy (`COLI_TEMP=0`), identical token streams (sha256 `f53f3896…` both runs).
- **Router fidelity: PASS** — engine STATS histogram over 191,196 selections, 128 distinct experts: **top-25% share = 80.3%** vs generator MC target 81.1%. The Zipf calibration survives the real sigmoid+bias top-k path, so cache-hit-rate findings transfer.
- **KV8/KV_TQ4 outputs differ from f32** — expected: these are the engine's opt-in, clearly-labeled lossy knobs; same-config reruns stay bit-identical.

---

## 1. Audit: what the brief assumed vs. what v2.0.0 already has

The brief's premises were written against an older tree. Current state, verified in source:

| Brief's premise | v2.0.0 reality (source-verified) | Status |
|---|---|---|
| "fp32 KV cache, 3.2× RAM overhead vs llama.cpp f16 KV" | MLA latent KV is 57× smaller by design; `KV8` (fp8+per-row scale, 3.9×) and `KV_TQ` (2–6-bit PolarQuant/rotated-int4, up to 7.6×) exist; `.coli_kv` persists across restarts | **Already shipped** — remaining gap: adaptive selection + GPU-native kernels |
| "`kvb_all` materialization is the biggest avoidable RAM consumer" | Fixed in #768: `KVB_FLASH_MB=2048` ceiling, `KVB_TILE_MB=512` flash-style tiling with online softmax (DSA rows exempt); `cap_for_ram` no longer reserves it permanently | **Already shipped** — remaining gap: tile the DSA-selected path too |
| "Fix MLOCK=1 + COLI_MMAP wiring 0 bytes / VRAM pages" | Found & fixed: the 363 GB-instead-of-231 GB VRAM-tier wiring bug is documented and guarded in `colibri.c` (qt_wire_split, cuda_eligible checks) | **Already fixed** — needs a regression test (mlock ulimit blocked the harness) |
| "v1.5.0 regression: 13-pt hit-rate loss, ~18% decode slowdown" | Real; fixed in v1.6.0 (#869 planner widest-width pricing, #914 pin budgets, prefill batch-union) | **Already fixed** — harness reproduces the mechanism (below) |
| "`TRUNK_RESIDENT_LAYERS=0` + `COLI_MMAP=1` zero-weight-cache mode, extend to all families" | Exists on the GLM engine (colibri.c) and K3_MMAP on kimi_k3; docs table: qwen36/inkling/deepseek_v41/olmoe are "Resident" only | **Partially shipped** — real open item |
| "mincore()/page-cache-aware reads" | Precedent exists: `.coli_ssd` probe uses mincore cold-range steering; qwen36 steers reads to mincore-cold ranges | **Partially shipped** — generalization open |
| "PILOT=1 prefetch, aggregate when it doesn't raise peak RAM" | `PILOT`, `PILOT_REAL`, `PILOT_EVICT_GUARD`, `COUPLE` all exist; 71.6% one-layer-ahead predictability measured upstream | **Already shipped** |
| "Cross-invocation persistent expert cache" | `.coli_usage` history + `AUTOPIN` + `PIN=auto` + mirror planner (`coli mirror plan/stage`) | **Already shipped** — a serialized warm-start slab map is the delta |

**Consequence:** the highest-leverage *new* work is not "add KV quantization" (it exists) but (1) making it adaptive, (2) porting it to the GPU paths, (3) extending trunk offload beyond GLM, and (4) formalizing the cache manager. The harness below measures each lever's effect size.

---

## 2. Measured results (synthetic harness, all numbers real)

### 2.1 Core matrix — synth_std, 448-token REPLAY decode, NVMe-class throttle

| Arm | tok/s | hit % | peak RSS | GB read | Notes |
|---|---|---|---|---|---|
| cap8, DIRECT, PIPE=0 | **93.5** | 64.5 | 195 MB | 0.64 | cold-ish baseline |
| cap8, DIRECT, **PIPE=1** | **104.3** (+11.5%) | 64.5 | 194 MB | 0.64 | I/O overlap pays |
| cap8, **URING=1** | 115.4 (+23%) | 64.5 | 196 MB | 0.64 | unthrottled (shim bypass) — real-disk A/B only |
| **cap64**, DIRECT, PIPE=1 | **129.6 (+39%)** | **97.5** | 911 MB | 0.24 | cache size dominates |
| cap8, buffered (page-cache warm) | 119.9 | 64.5 | 195 MB | — | warm page cache = RAM-speed reads |
| cap8, **SATA emul** (500 MB/s, 150 µs) | 104.6 | 64.5 | 194 MB | 0.64 | slower disk, same overlap benefit |

**Read:** with a real-disk-shaped I/O path, per-layer LRU capacity is worth ~+39% (64.5→97.5% hit); read/compute overlap (PIPE=1) is worth ~+11.5% at constant hit rate. This is the causal shape behind the v1.5.0-class regressions: anything that silently shrinks effective cache costs tens of percent of decode speed.

### 2.2 Learned pins (research area #3) — the self-tuning cache

Seed `.coli_usage` via 12 SERVE turns (57,744 selections), then 512-token REPLAY at cap=8:

| Arm | tok/s | hit % |
|---|---|---|
| cold (history moved aside) | 99.3 | 64.5 |
| **PIN=auto** (383 experts, 0.6 GB warm) | **139.4 (+40.3%)** | **88.5 (+24.0 pts)** |

The engine's own pin placement: `[PIN] placement: 0 VRAM + 383 RAM expert (0.6 GB warm)`. **This quantifies the payoff of history-driven placement on a Zipf workload** — and the cost of losing it (the v1.5.0 regression shape, +13 pts hit ≈ tens of percent tok/s).

### 2.3 KV cache RAM at CTX=131072 (research area #1) — SERVE mode, pool pre-allocated

| Arm | projected KV pool | peak RSS | tok/s (n=2, noisy) | hit % |
|---|---|---|---|---|
| f32 | **1.66 GB** | 373 MB | 38.8 / 49.5 | 92.7 / 93.0 |
| KV8 | **0.43 GB (−74%)** | 372 MB | 44.3 / 44.9 | 93.5 |
| KV_TQ=4 | **0.22 GB (−87%)** | 361 MB | 46.3 / 47.1 | 93.5 |

KV8/TQ4 show **no throughput penalty here** (freed budget → slightly higher hit), but note f32 run-to-run variance is large at n=2 — treat directionally. Outputs differ from f32 (lossy knobs, opt-in; default path untouched).

### 2.4 Dense-trunk offload (research area #2) — synth_huge_trunk, RAM_GB=2

| Arm | peak RSS | rc |
|---|---|---|
| trunk resident (default) | 878 MB | 0 |
| TRUNK_RESIDENT_LAYERS=0 + COLI_MMAP=1 | 1,843 MB | 0 |
| … + PIPE=1 | 1,848 MB | 0 |

**Honest reading:** the mmap arm *raises* RSS on this host — mapped-and-touched pages count in RSS and the freed budget flows to the expert LRU. The real benefit is *evictability under pressure* (file-backed pages reclaim; slab pages don't), which this 16 GB box cannot demonstrate. **Requires real-model validation on a RAM-scarce host.** No OOM/failure in either arm; fallback behavior correct.

---

## 3. Prioritized roadmap

### Tier 1 — Immediate wins (low effort, high impact)

**1.1 Adaptive KV quantization policy (`KV_AUTO`)**
- **Description:** auto-select `KV8` (or `KV_TQ=4` where row shapes qualify) when the projected KV pool exceeds a fraction of the RAM budget, with a one-line notice; keep lossless f32 as default and the auto path clearly labeled "changes precision".
- **Harness result:** pool 1.66 → 0.43 GB (−74%) at CTX 131k with no measured decode penalty; hit rate +0.8 pt.
- **Expected real-model impact (extrapolation, medium-high confidence):** GLM-5.2 (79 layers, kv_lora 512 + qk_rope 64 → 2,304 B/token/layer) projects a **~47.7 GB f32 pool at 256k** (formula from `kv_pool_bytes`, verified against the engine's own projection line). KV8 cuts that to ~12 GB, freeing **~35 GB ≈ 1,900 int4 experts ≈ 10% of the routed set** for the cache — worth several tok/s points in cold streaming, matching upstream's "35 GB/slot at 256k" note. Assumption: hit-rate→tok/s slope from §2.1 transfers.
- **Complexity:** 1–2 person-weeks (policy + notice + docs; codecs exist).
- **Risks:** quality regression (opt-in/label; needs per-model perplexity gate), interaction with `COLI_CUDA_PIPE=0` forcing.

**1.2 Ship the I/O-fidelity layer as a research tool + `iobench` integration**
- **Description:** the `LD_PRELOAD` throttle (with its documented io_uring blind spot) plus a `--profile` flag reproducing NVMe/SATA classes lets any contributor A/B cache policy without a 372 GB download.
- **Harness result:** throttle verified to 1% accuracy; discovered that **`DIRECT=0` runs measure the page cache, not the disk** (119.9 vs 93.5 tok/s) — a trap the project's own benchmarking doc warns about, now mechanically enforceable.
- **Expected real-model impact:** high value for contributor ramp-up; no runtime effect.
- **Complexity:** <1 week. **Risks:** none (research-only path).

**1.3 CAP_RAISE visibility for A/B hygiene**
- **Description:** the matrix's first pass was invalid because default `CAP_RAISE=1` silently re-grew requested caps ("cap raised 64→128"). Add a `PROF`-visible one-liner whenever the cap is auto-raised beyond the requested value (it logs to stderr already; make it a `[PROF]` field so saved reports carry it).
- **Harness result:** first matrix measured the same warm point 7×; with `CAP_RAISE=0` the cache-size effect (+39%) appeared.
- **Complexity:** days. **Risks:** none.

### Tier 2 — Medium-term projects (significant refactoring)

**2.1 Unified, feedback-driven cache manager**
- **Description:** today pin budget (confidence formula), LRU cap (`CAP_RAISE`), `RSS_GUARD_GB`, and the usage history interact through heuristics. Replace with one controller that measures live hit rate and adjusts pin/LRU split (the `REPIN` RFC is the seam). 
- **Harness result:** the prize is measured — pins recover **+24 hit points / +40% tok/s** on a Zipf stream; mis-sizing costs the same (§2.1).
- **Expected real-model impact (high confidence):** same mechanism at GLM scale; guard against the documented overfit-to-prompt failure mode with the confidence cap already present.
- **Complexity:** 3–4 person-weeks. **Risks:** oscillation; cold-start transient (already characterized as uncharacterized for `DEGRADE_ZERO`).

**2.2 Trunk offload for the remaining families + auto-detect**
- **Description:** extend `TRUNK_RESIDENT_LAYERS`-style mapped trunk to qwen36/inkling/deepseek_v41/olmoe (currently resident-only), auto-engage when trunk > RAM budget with a warning.
- **Harness result:** GLM arm runs correctly (no OOM); RSS metric confounded by page cache — **requires real-model validation** on a memory-scarce host (flagged).
- **Expected real-model impact (medium confidence):** makes mid-size dense checkpoints (e.g. Qwen3.8-27B trunk, 54 GB f16) bootable on <32 GB hosts at all, per the docs' own "this is how a model runs at all, not fast" framing.
- **Complexity:** ~3–4 weeks/engine. **Risks:** page-cache thrash, kernel eviction of hot pages, NUMA placement.

**2.3 GPU-native quantized-KV kernels**
- **Description:** `KV8`/`KV_TQ` currently force `COLI_CUDA_PIPE=0` (CPU attention fallback, "native kernels are follow-up PRs" per ENVIRONMENT.md). Port the quantized-KV decode to CUDA/Metal/Vulkan.
- **Harness result:** N/A on this CPU-only box — the CPU-path pool win (−74/−87%) is measured; the GPU pipeline loss is documented upstream.
- **Expected real-model impact (medium confidence):** removes the current trade-off, i.e. long-context GPU decode keeps the pipeline AND the small pool.
- **Complexity:** 4–6 weeks. **Risks:** precision on tensor-core paths; per-vendor validation.

**2.4 Tile the DSA-selected attention path**
- **Description:** `kvb` flash tiling explicitly exempts DSA-selected rows ("top-keep list is not tileable by t-range"). A two-band tiler or a selection-aware gather would remove the last unbounded prefill transient.
- **Harness result:** not measurable on synth (index_topk=4096 disables sparsity); **requires real-model validation**.
- **Complexity:** 2–3 weeks. **Risks:** output divergence class (same as existing tiling, documented).

### Tier 3 — Long-term research (moonshots)

**3.1 Page-cache-aware expert cache (mincore residency steering)**
- Extend the existing mincore cold-range steering (`.coli_ssd` probe, qwen36 read steering) into LRU *admission*: prefer admitting experts whose pages are already resident; demote those the kernel holds but never re-hits. Harness can't test honestly without a droppable cache (root) — **requires real-hardware validation**.
- Complexity: 4–8 weeks. Risks: `mincore` raciness, portability (Windows has no equivalent).

**3.2 `MADV_PAGEOUT` proactive eviction**
- Proactively page out cold mapped trunk/expert ranges instead of waiting for pressure; pairs with 3.1. Same validation constraint.

**3.3 Persistent warm-start slab map**
- `.coli_usage` + `PIN=auto` already persist *which* experts matter; persist *the slabs themselves* (mmap-backed store + serialized pin map) so a restart begins at hit-rate ~88.5% with zero cold reads. Harness demonstrated the hit-rate target; the persistence format is the work.

**3.4 FUSE/loopback mirror emulator**
- Emulate `COLI_MODEL_MIRROR` multi-SSD setups with per-loopback bandwidth caps (the throttle layer already does this for a single dir; a FUSE passthrough with two mountpoints is the extension) to test weighted striping without two physical drives.

---

## 4. Final constraint checklist

- ✅ **Pure C, zero dependencies**: the engine was built unmodified from source; every experiment is external (env vars + `LD_PRELOAD`). No engine code was changed, so no dependency was introduced.
- ✅ **No real model downloads**: all containers are synthetic (generator is numpy-only; largest is 4.6 GB of random int4). The 372 GB GLM-5.2 container was never touched.
- ✅ **Every speedup/RAM saving measured or labeled**: §2 numbers are harness-measured (with n and caveats); §3 real-model numbers are labeled extrapolations with the formula and assumptions stated.
- ✅ **Lossless by default**: all experiments keep the engine's defaults; every lossy knob (`KV8`, `KV_TQ`, `DEGRADE_ZERO`, `CACHE_ROUTE`) is the engine's own opt-in and was used only in labeled arms. Determinism gate PASSES on the default path.

## 5. Reproduction

```bash
# one-time: build engine + harness (WSL2 Ubuntu, gcc 15)
cd ~/colibri/c && make colibri
cd ~/colibri-research/harness && gcc -O2 -shared -fPIC iothrottle.c -o iothrottle.so -ldl

# generate models (deterministic seeds)
python3 ~/colibri-research/synthgen/make_synthetic_model.py --out ~/synth_models/synth_std --seed 7
python3 ~/colibri-research/harness/make_tokenizer.py ~/synth_models/synth_std/tokenizer.json 2047

# full matrix + experiments
python3 run_workloads.py --model ~/synth_models/synth_std --throttle-so $PWD/iothrottle.so
python3 exp_kv_serve.py && python3 exp_kv_and_trunk.py && python3 exp_pin_recovery.py
```

Raw JSON results: `~/colibri-research/harness/results_{std,kv_serve,kv_trunk,pin}.json` (copies in `C:\Users\dylan\projects\colibri-harness\`).
