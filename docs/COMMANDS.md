# Bonsai command reference

Every command the stack exposes, in copy-paste form. For installation see
[INSTALL.md](INSTALL.md); for the measured numbers behind the knobs see the
reports in `docs/`. Lines starting `#` are comments; `<angle>` tokens are
yours to fill.

---

## 1. Quick map

| You want to… | Use |
|---|---|
| install from scratch | [INSTALL.md](INSTALL.md) |
| **benchmark a dense 27B (Qwen3.8-27B class), start to finish** | [BENCH_DENSE27B.md](BENCH_DENSE27B.md) |
| chat / serve / benchmark a model | §2 the `q36-minipc` launcher |
| drive the engine binary directly, custom knobs | §3–4 |
| build an agent client against the server | §5 wire protocol |
| create a test container without downloads | §6 generators |
| reproduce the experiments / measurements | §7 drivers |
| rebuild the engine patch stack from stock | §8 patch scripts |
| verify an install is sound | §9 parity gates |

---

## 2. The `q36-minipc` launcher (start here)

One command composes the whole profile — it auto-detects MoE vs dense from
`qwen36_meta.json`, picks the cache size, threads, context, KV tier and
neuron fraction, and hands the engine a verified env.

```bash
q36-minipc <command> <model_dir> [arg]
```

| Command | What it does | Notes |
|---|---|---|
| `serve <model>` | OpenAI-compatible server (`SERVE=1 OPENAI=1`) | one slot per conversation: set `Q36_SLOTS` |
| `chat <model>` | interactive chat in the terminal (512 tokens, temp 0.7) | |
| `bench <model> [prompt]` | honest battery: phase-split timers, peak RSS, expert hit rate | prints `decode tok/s = 1000/(TOTAL ms/token)` — the `Speed:` line includes prefill, never quote it as decode |
| `seed <model>` | (MoE) write a Zipf-router `.coli_usage` prior | makes expert-cache benchmarks face realistic traffic |
| `calibrate <model>` | (dense) one `HOT_PROFILE` pass; set `Q36_CORPUS=<text file>` | produces `<model>/hot_profile.json`. Full-compute pass: ETA banner printed; corpus capped at `Q36_CAL_TOKENS` (default 1024, `0`=whole file); engine stderr streams live; **profile written only at clean exit — don't ^C mid-pass** |
| `sort <model> [profile]` | (dense) hotness-sorted copy of the container via `hot_sort.py` | → `<model>-sorted`; GRIFFIN makes this optional |
| `doctor <model>` | report: engine binary, knob reference counts, model class, resolved level | run this first whenever something behaves oddly |
| `watch` | live-render `[prefill]` progress lines from `/tmp/calib.err` as a one-line bar (`[calib |####......|  42%  t=..s eta=..s]`) | second terminal beside a running `calibrate`/`bench`; `calibrate`/`bench` also render inline |

### 2.1 Levels (`Q36_LEVEL`)

| Level | expert cap | threads | context | KV tier | hot frac | for |
|---|---|---|---|---|---|---|
| `eco` | 24 | 2 | 4096 | kv8 | 0.25 | smallest RAM (measured ~4.8 GB, 96.8% hit) |
| `balanced` *(default)* | 48 | 4 | 8192 | kv8 | 0.25 | the mini-PC daily driver |
| `quality` | 64 | 4 | 16384 | f32 | 0.5 | RAM-rich boxes |

### 2.2 Launcher environment knobs

| Knob | Values | Meaning |
|---|---|---|
| `Q36_LEVEL` | `eco` `balanced` `quality` | the preset table above |
| `Q36_KVQ` | `f32` `kv8` *(default)* `tq4` `tq3` | KV cache tier — see `docs/KVTQ_QWEN36.md` for the measured RAM/speed trade |
| `Q36_HOT_FRAC` | 0.25 … 1.0 | dense only: fraction of MLP neurons computed; **1.0 = exact full model** |
| `Q36_GRIFFIN` | `0`/`1` (default 1) | dense only: per-prompt neuron selection, works on unmodified containers |
| `Q36_HOT_LAYERS` | JSON file | per-layer `{"L0":0.5,…}` fractions (falls back to the global frac) |
| `Q36_CORPUS` | text file | the calibration corpus for `calibrate` |
| `Q36_CAL_TOKENS` | integer (default 1024; `0` = whole corpus) | cap the calibration prefill (~4 chars/token); the energy ranking converges early — bound it on small boxes |
| `Q36_PROGRESS` | `1`/`0` (default 1) | `[prefill] p=.. layer=.. t=.. eta=..` stderr line per layer of every multi-row forward (the progress-bar feed); stderr-only, compute untouched; also makes a killed `calibrate` keep a partial profile (periodic save every 8 layers) |
| `Q36_SORT_OUT` | dir | output dir for `sort` |
| `Q36_RAM_GB` | integer (default: auto) | tell the launcher the real RAM; caps the expert cache accordingly |
| `Q36_SLOTS` | integer (default 1) | conversations the server multiplexes (`KV_SLOTS`); **one per concurrent conversation** |
| `Q36_ENGINE` | path | override the engine location (default `~/colibri/c/qwen36`) |

### 2.3 The two day-one recipes

**MoE model (Qwen3-Coder-30B-A3B class):**

```bash
q36-minipc seed  ~/Models/qwen36_i4_gs64     # once
q36-minipc serve ~/Models/qwen36_i4_gs64      # then daily; Q36_LEVEL=eco on 8 GB
```

**Dense model (27B class) — stock loads OOM on 8 GB; this is the Bonsai path:**

```bash
Q36_CORPUS=~/my_corpus.txt q36-minipc calibrate ~/Models/qwen27_i4   # once
# calibrate is a FULL-COMPUTE pass (HOT_FRAC=1.0 by necessity): expect ~1-2 s/token
# on a dense 27B at mini-PC core counts. The launcher caps the corpus at
# Q36_CAL_TOKENS (default 1024 ≈ 20-35 min) and prints an ETA first; logs stream live.
q36-minipc sort ~/Models/qwen27_i4                                   # once
q36-minipc serve ~/Models/qwen27_i4-sorted
# or skip both steps: GRIFFIN adapts per-prompt on the UNMODIFIED container
```

---

## 3. The engine binary directly (advanced)

The launcher is a thin env-composer; underneath sits one binary:

```bash
~/colibri/c/qwen36 <cap> <threads> <prompt_file>
#     cap    = expert-cache slots per layer (dense: any value, unused)
#     threads= OpenMP threads
#     prompt = a text file to generate from, or ref.json for oracle mode
```

- **Oracle mode**: a `.json` as `<prompt_file>` runs the reference ids and
  exits 1 on any mismatch — by design; it is the correctness harness.
- **Server mode**: `SERVE=1` (see §5); no prompt file needed.
- Generated-token count: **`N_NEW`** (default 64). `NGEN` does nothing.
- Greedy: `COLI_TEMP=0`.
- Read the stderr banner before trusting any run — the resolved config line
  (`== qwen36 Phase-2 engine | cache=… ctx=… ==`), `[enc] prompt tokens: N`,
  `[qwen36] KV_TQ=…` / `KV8=…`, `TTFT: …`, `[timers] decode: N tokens …`.

Minimal example, no launcher:

```bash
SNAP=~/synth_models/synth_qwen27 Q36_MAXT=2048 N_NEW=24 COLI_TEMP=0 \
  OMP_NUM_THREADS=4 ~/colibri/c/qwen36 32 4 ~/my_prompt.txt
```

### 3.1 Engine environment knobs (reference)

**Model & loading**

| Knob | Meaning |
|---|---|
| `SNAP=<dir>` | container directory (the model) |
| `Q36_MAXT=<n>` | context ceiling; **must cover prompt + reply** |
| `OMP_NUM_THREADS=<n>` | thread count (launcher sets it with `COLI_NO_OMP_TUNE=1`) |
| `COLI_DENSE_BITS=<4\|8\|16>` | dense-weight quantization at load |
| `COLI_KEEP_F32=1` | keep the f32 copy beside the quantized one (debug) |

**KV cache tiers** (mutually exclusive; `KV_TQ` wins)

| Knob | Meaning |
|---|---|
| `KV8=1` | fp8-e4m3 KV rows + per-row f32 scale (−16% RSS at 8k ctx) |
| `KV_TQ=<2..8>` | packed KV rows; `KV_TQ_CODEC=1` rot-int4 (default), `0` PolarQuant |
| `Q36_KVQ` | the launcher's name for the same choice |
| `Q36_FLASH=0` | disable flash prefill (it is **on by default**) |

**Dense paging** (the 27B-on-8GB machinery)

| Knob | Meaning |
|---|---|
| `TRUNK_MAP=1` | mmap the dense trunk; resident RSS = KV + norms only |
| `GRIFFIN=1` | per-prompt neuron selection (training/calibration-free) |
| `HOT_FRAC=<f>` | compute only the top fraction of neurons (1.0 = exact) |
| `HOT_PROFILE=<file>` | one calibration pass → `hot_profile.json` |
| `HOT_LAYERS=<file>` | per-layer fraction table |
| `TRUNK_WARM=1` | one `madvise(WILLNEED)` sweep at load (prefix-only at frac<1) |

**MoE expert cache**

| Knob | Meaning |
|---|---|
| `PIN=auto` | learned expert pins from the usage history |
| `PILOT=2` | 2-slot prefetch pilot (free when warm) |
| `WIDE=<n>` | wide-knob expert grouping |
| `CAP_RAISE=0` | **always set on small boxes** — stops the cache silently re-growing past RAM (the launcher does it for you) |
| `RAM_GB=<n>` | RAM ceiling for cache sizing |
| `USAGE_SAVE=1`, `COLI_USAGE=<file>` | expert usage history (`.coli_usage`) read/write |

**Prefix reuse & speculation** (the agent-facing machinery)

| Knob | Meaning |
|---|---|
| `COLI_KV_PREFIX=0` | disable transcript-prefix reuse (A/B hatch) |
| `COLI_PREFIX_LOG=1` | print the reuse decision per turn — **always verify with this** |
| `COLI_LOOKUP=0` | disable prompt-lookup drafts (on by default) |
| `COLI_LOOKUP_DRAFTS=<1..5>` | max drafts per verify |
| `COLI_SPEC_GATE=0` | draft every proposal in full (tests) |
| `COLI_LOOKUP_FORCE=accept\|mixed\|cycle\|rowN` | oracle-driven drafting for tests |
| `Q36_FAST_SOFTMAX=1` | AVX2 fast exponential in the attention softmax (−6.5% attention ms/token at 16k ctx, TTFT-neutral; lossy-by-rounding ~2.5e-7, opt-in; needs an AVX2+FMA build) |

**Measurement**

| Knob | Meaning |
|---|---|
| `COLI_TIMERS=1` | `[timers] decode: N tokens … TOTAL ms/token` — the honest decode rate |
| `Q36_FLASH_DEBUG` | flash tile decisions |
| `ENC_DEBUG=1` | dump the encoded prompt ids |

---

## 4. What the log lines mean

| Line | Read it as |
|---|---|
| `== qwen36 Phase-2 engine \| cache=48/layer bits=4 ctx=8192 … ==` | the resolved config — **read before trusting a run** |
| `[enc] prompt tokens: N` | true prompt length (byte-level tokenizer: ≈ bytes, not words) |
| `[qwen36] KV_TQ=4 codec=1 …: KV rows packed 64 B …` | KV tier + packed row width vs f32 |
| `TTFT: 12.3 s` | prefill cost |
| `[timers] decode: 47 tokens … TOTAL 21.3 ms/token` | decode rate = 1000/21.3 = 46.9 tok/s |
| `Speed: 3.5 tok/s` | total wall time incl. prefill — never a decode number |
| `[PREFIX] reusing 1608 of 1736 prompt tokens (93%)` | transcript reuse hit |
| `Expert cache hit rate: 99.6%` | MoE cache effectiveness |

---

## 5. The serve wire protocol (agent builders)

With `SERVE=1` (no `OPENAI=1`) the engine speaks a line protocol on
stdin/stdout — this is what the prefix-reuse battery drives, and the lowest-
friction way to embed it. Wait for the `READY` sentinel first.

```text
→ SUBMIT <id> <slot> <plen> <max_tok> <temp> <top_p>\n
→ <exactly plen payload bytes>\n
← ACCEPT <id> <prompt_tokens>\n
← DATA <id> <n>\n<n bytes>\n            (repeated, UTF-8 chunks)
← DONE <id> STAT <gen> <tps> <hit%> <rss_gb> <np> <limited>\n
→ CANCEL <id>\n                          (abort the current turn)
```

Rules that matter for reuse (measured in `docs/PREFIX_REUSE_QWEN36.md`):

1. **Resend history byte-for-byte, append-only** — any edit diverges the
   whole prefix and re-prefills everything.
2. **One slot per conversation** (`KV_SLOTS`/`Q36_SLOTS`); interleaving two
   conversations on one slot evicts each other's state.
3. `Q36_MAXT` must cover the whole transcript, or the reuse record drops.

---

## 6. Synthetic containers (zero download)

All generators are numpy-only, fixed-seed, and write to `~/synth_models/`.
Same geometry as the real models, random weights — this is what every
published number in the repo was measured on.

```bash
# small dense student (~1–2 GB) — the fast smoke test (needs TRUNK_MAP=1, i.e. the launcher)
python3 scripts/gen_student_i4.py

# full-shape Qwen3-Coder-30B-A3B MoE geometry (48L × 128 experts, int4-gs64)
python3 scripts/gen_coder30_synth.py

# full 27B dense shape in f16 (~49 GB, for TRUNK_MAP experiments)
python3 scripts/gen_qwen38_27b_f16.py

# minimal byte-level tokenizer where a container lacks one
python3 scripts/make_tokenizer.py ~/synth_models/<name>/tokenizer.json <eos_id>

# Phase 2 (in development): add a synthetic MTP head to any container
python3 scripts/gen_mtp_head_synth.py ~/synth_models/synth_coder30
```

`gen_qwen27_synth.sh` builds the 27B-shaped `synth_qwen27` (365 MB, the
battery container) but expects the private harness helper
(`make_synthetic_model.py`) to be on its path — see §8's note on
reproducibility if it is missing in your tree.

---

## 7. Experiment drivers (`scripts/`)

Each writes a JSON into `~/colibri-research/harness/` (WSL home, never
`/tmp` — each `wsl.exe` invocation gets a private `/tmp`).

| Script | Measures | Arms |
|---|---|---|
| `exp_kvtq_q36.py [arms…]` | KV tiers: RSS / TTFT / honest decode (`COLI_TIMERS`) | `A_f32 B_kv8 E_tq4_c1 F_tq4_c0 G_tq3_c0 H_tq4c1_i4dense J_f32_flash I_tq4c1_flash` |
| `exp_prefix_q36.py [arms…]` | transcript-prefix reuse through the serve protocol | `mux_on mux_off single_on single_off mux_agent` |
| `exp_qwen27_final.py` | the original KV8 low-RAM matrix | A–D |
| `exp_kv_serve.py`, `exp_kv_and_trunk.py` | KV pool size / trunk offload | |
| `exp_pin_recovery.py` | learned pins | |
| `run_workloads.py`, `verify_structural.py`, `diag_decode_scaling.py` | cold/steady matrix, structural gate, scaling | |

All accept `--snap --maxt --nnew --out --prompt` style overrides where
relevant; scrub inherited env per arm (the scripts do it themselves).

Example:

```bash
python3 scripts/exp_kvtq_q36.py A_f32 E_tq4_c1      # two arms, appends to the results JSON
```

---

## 8. Rebuilding the engine from stock (patch scripts)

`engine/` is a snapshot; the patch chain below reproduces it **byte-exactly**
from stock colibri `qwen36.c` (clone github.com/JustVugg/colibri @ `bf24429`,
then `cp engine/*.c engine/*.h c/`, `make -C c qwen36` — full detail in
[INSTALL.md](INSTALL.md)). Verified end-to-end by `tools/verify_engine.sh`
and CI on every push:

| Order | Script | Applies |
|---|---|---|
| 1 | `patch_kv8_flash_v1.py` | KV8 tier (both parts) + flash-tile prefill v1 — regenerated (E1) |
| 2 | `patch_trunkmap.py` | mmap dense trunk (f16 stage) |
| 3 | `patch_trunkmap_v2.py` | trunk int4 mapping |
| 4 | `patch_hotstack_v1.py` | hot-neuron paging + self-calibration + GRIFFIN + mapfast/pagebatch — regenerated (E1) |
| 5 | `patch_kvtq.py` | KV_TQ tier (PolarQuant / rot-int4) |
| 6 | `patch_softmax.py` | Q36_FAST_SOFTMAX (AVX2 fast exp in softmax) |

Each script refuses to run twice and takes an optional path argument
(default `~/colibri/c/qwen36.c`); backups `.orig/.premap/.prehot/.pretq/.presm`
beside the source are the checkpoint md5s in `tools/patch_manifest.json`.
The two regenerated patchers replace historical per-feature scripts that had
drifted (KV8 part-1 never committed; flash v2 vs shipped v1; interleaved
hot/griffin/mapfast/pagebatch hunks) — superseded files and the full story in
`tools/attic/ATTIC.md`; audit record in `docs/ROADMAP.md` E1 (CLOSED).

Other tools: `hot_sort.py <model> [profile]` (hotness-sorted container copy),
`hotcal_pipeline.py`, `seed_zipf_usage.py` (calibration + Zipf usage prior).

---

## 9. Verify an install (parity gates)

They are pure C99, no model needed. Since `engine/kv_fp8.h` is vendored
(byte-identical from colibri `bf24429`), they compile from the Bonsai tree alone:

```bash
gcc -O2 -I"$HOME/bonsai/engine" -o /tmp/kvtq "$HOME/bonsai/engine/kv_tq_parity_q36.c" -lm && /tmp/kvtq
# PARITY OK: FWHT, radius, direction, linearity, inert, accounting — all gates passed

gcc -O2 -I"$HOME/bonsai/engine" -o /tmp/kv8p  "$HOME/bonsai/engine/kv8_parity.c"        -lm && /tmp/kv8p
# all 256 codes identical … PASS

gcc -O2 -mavx2 -mfma -I"$HOME/bonsai/engine" -o /tmp/smb "$HOME/bonsai/engine/softmaxbench_q36.c" -lm && /tmp/smb
# fast-exp gates: rel err 2.46e-07, underflow zone safe, more accurate than stock
```

(From a colibri checkout the same commands work with `-I"$HOME/colibri/c"` on the
copies in `c/`.) Snapshot-chain check, no compiler needed: `bash tools/verify_engine.sh`.


Then a 5-minute smoke test end-to-end:

```bash
python3 scripts/gen_student_i4.py
q36-minipc doctor ~/synth_models/student_1b_i4
q36-minipc bench   ~/synth_models/student_1b_i4
```

---

## 10. Phase 2 note (MTP head)

**Part 1 LANDED (2026-10-19, `test` branch):** `tools/patch_mtp.py` wires the
loader + knobs — `Q36_MTP=1` (refuses headless containers and MoE heads),
`Q36_MTP_DRAFTS=0..3` (default 2, parsed + range-checked), `Q36_MTP_FORCE=<str>`
(parsed; the oracle comes with part 2). The dense head (norms, qk-normed full
attention, `mlp.{gate,up,down}`) loads f32-resident from
`model-mtp.safetensors`; report line `[qwen36 MTP] head loaded: …`. Part 1 is
inert by construction: knob-off runs are byte-identical, knob-on runs differ
only by the report line — both enforced in CI (`engine-build.yml`).
**Part 2 LANDED (same branch):** dense-head drafting + verify —
`tools/patch_mtp_draft.py`: head forward in the S=1 dialect (own K/V rows, qk-norm,
partial rope, GQA, output gate), hidden capture at the lm_head seam, proposals
beside lookup (`spec_gate_pick` values decide), head-row rollback on rejection,
`[qwen36 MTP] run:` acceptance report. With random synthetic weights acceptance
is ~0 by construction — the CI gate is that greedy output stays byte-identical
to stock for `Q36_MTP_DRAFTS` 0..3 (verify exactness + rollback proof).
**Part 3a LANDED (same branch):** `Q36_MTP_FORCE=accept|mixed|cycle|rowN` drives
the MTP proposal path from ref.json's oracle ids; `DUMP_IDS=<path>` writes the
generated id sequence (CI bootstraps ref.json with it). CI gates: forced-accept
⇒ 100% acceptance AND the id sequence equals the stock run's (the design's
"reproduces the oracle text exactly"); mixed/cycle ⇒ sequence still exact under
rejections + head-row rollback. Note: the *final logits* can legitimately differ
from a plain step's when the last pick is answered from a verify batch row (S>1
fp order) — pre-existing engine behavior shared with accepted lookup drafts,
which is why the oracle invariant is sequence identity, not logit bits.
**Part 3b (open):** MoE head (resident int4 experts), draft-cost battery on
real hardware → `docs/MTP_QWEN36.md`.
*Everything here is synthetic-container measured; real-checkpoint validation
is the declared open step (README *Status*). If a command misbehaves, run
`q36-minipc doctor` and read the stderr banner before anything else.*
