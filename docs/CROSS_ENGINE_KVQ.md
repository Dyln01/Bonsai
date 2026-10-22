# Cross-engine KV8/KV_TQ ports (Phase 4c) — design contract

**Date:** 2026-10-22 · status: **slice 1 shipped (build proof + CI-sized qwen38 container)**
**Order:** qwen38 first (GQA sibling: same `st.h`, spec machinery, chain idioms,
and an in-repo generator), glm53 second (MLA: latent KV makes TQ absorption
intrinsic — the codec rides ~1/8 the rows of a GQA cache).

## What travels unchanged

- `engine/kv_tq.h` codecs (PolarQuant / rotated-int4) — written engine-agnostic;
  the 3b-rot absorbed-rotation pattern (rotate queries once, dot raw rotated
  levels, accumulate rotated, unrotate once) applies verbatim to any GQA decode.
- `engine/kv8_fused.h` + the scale-hoisting contract (radius/scale in the same
  per-row slot, `dequant(r=1)*s == dequant(s)`).
- The parity gates (`kv8_parity.c`, `kv_tq_parity_q36.c`, `tq_absorb_parity.c`)
  compile against any engine that overlays the headers — codec math is engine-free.
- Methodology: `DUMP_IDS`/`DUMP` identity and cosine gates, honest `[timers]`
  decode numbers, `run_battery.sh` (its `Q36_ENGINE` env already parameterizes
  the binary; per-engine prompt/cap args documented per slice).

## What is engine-specific (per-engine patchers, own manifest section)

KV pool layout and growth path, decode-attention anchors (qwen38: GQA like
qwen36; glm53: MLA latent + decode-absorb already), knob names/parse sites,
flash/prefill tile loops, report lines. Each gets its own patcher
(`tools/patch_kv8_q38.py`, `tools/patch_tq_q38.py`, `tools/patch_kvq_glm53.py`),
its own backup suffix (`.q38prekv8` …), and a **separate `cross_engine` section
in `tools/patch_manifest.json`** — never interleaved with the qwen36 chain,
which stays the source of truth for this repo's engine snapshot.

Upstream drift policy: patchers anchor to colibri **bf24429** sources; the CI
build-proof step (slice 1) recompiles both siblings on every push, so an
upstream anchor break surfaces as a red build within minutes of a pin change,
and pins move only by explicit commit.

## Gates per slice

1. **Slice 1 (this commit):** `make qwen38 glm53` green in CI with Bonsai
   headers overlaid; `gen_qwen38_27b_f16.py` SYNTH_-parameterized small
   container generated in CI. No tier code yet.
2. **Slice 2 (qwen38 KV8):** codec parity unchanged; kv8-off runs byte-identical
   to stock qwen38 (`DUMP_IDS`); kv8-on vs off last-logit cosine ≥ 0.999 (lossy
   tier, band pre-registered); RSS delta matches the row-width math.
3. **Slice 3 (qwen38 KV_TQ + absorbed rotation):** same identity/cosine ladder
   per tier (tq4/tq3), plus `Q38_TQ_ABSORB` 0/1 id-identity (the absorb gate
   pattern from qwen36 3b), plus the sim/battery arms on the mini-PC.
4. **Slice 4 (glm53 latent TQ):** MLA latents are already absorbed; TQ on the
   latent rows must be decode-neutral-or-better (that is the GLM engine's
   documented KV8 property) — gate: cosine band + decode ms/tok non-regression
   at equal RSS, on a glm53 synthetic container (generator lands with the slice).
5. Every slice: qwen36 chain + all existing gates stay green (no cross-talk).

## Out of scope

deepseek-v4 engines, Vulkan/CUDA KV tiers, serve-mux KV-slot semantics per
engine (qwen36-only feature), real-weights runs (Phase 5 covers qwen36 first;
siblings get batteries only if their ports pass synthetic gates).

## Slices

- [x] 1 — build proof + parameterized qwen38 generator (CI green)
- [x] 2 — `tools/patch_kv8_q38.py` (12 anchored edits over qwen38.c +
      qwen38_core.h: knob, byte-wide pools + scales, quantize-on-write,
      dequant-once-per-selected-row in the QSA gather, prefix/serve refusals)
      + CI **compile gate** (stock build, patch, rebuild). Behavioral gates
      blocked on 2a: the in-repo qwen38 generator emits the qwen3_5 dialect while
      pinned qwen38 requires `qwen4_exp_text` (+ PLE/QSA config surface).
- [ ] 2a — qwen38-dialect synthetic container: reverse-engineer the
      qwen38_core.h parser (config: model_type/hidden_act/output_gate_type/
      attention_bias/tie_word_embeddings/rope_parameters + PLE & QSA indexer
      numerics + tensor naming incl. ple_* weights) and fix
      `gen_qwen38_27b_f16.py` (SYNTH_-parameterized already).
- [ ] 2b — behavioral gates: stock vs patched-KV8=0 `DUMP_IDS` identity;
      KV8=1 vs 0 last-logit cosine ≥ 0.999; RSS delta = row-width math.
- [ ] 3 — `patch_tq_q38.py` + absorb + gates
- [ ] 4 — glm53 latent TQ + generator + gates
