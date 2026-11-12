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
| shrink the dense trunk to int4 (offline) | §11 `tools/pack_trunk_i4.py` |
| know what a container will cost on THIS host | §12 `tools/project_q36.py` |

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
| `doctor <model>` | report: engine binary, knob reference counts, model class, trunk dialect, window eligibility, resolved level | run this first whenever something behaves oddly |
| `pack <model> [out]` | (dense) int4-pack the trunk offline via `pack_trunk_i4.py` §11 | the lever that also makes the window allocatable |
| `fracs <model> [out]` | (dense) per-layer `HOT_FRAC` schedule via `make_layer_fracs.py` §13 | writes `<model>/hot_layers.json`, auto-used |
| `project <model>` | what this container costs on THIS host, by class (`project_q36.py` §12) | `Q36_PROJ_STACK=1`, `Q36_WIN_MB=`, `Q36_LAYER_FRACS=` add the ladder |
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
| `Q36_CORPUS` | text file | the calibration corpus for `calibrate` — representative traffic samples; worked example in `docs/examples/q36_corpus_example.txt` and `docs/BENCH_DENSE27B.md` §calibration corpus |
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

### 7.1 The one-command battery (`tools/run_battery.sh`)

```bash
tools/run_battery.sh <model_dir> <prompt_file> [out_json]
```

Per-arm honest `[timers]` decode, TTFT, PEAK RSS and rc; writes `results/battery_<host>_<ts>.json`
plus per-arm logs beside it. Env that matters on an HDD box: `BATTERY_ARMS` (subset, e.g.
`"A W0 W256"`), `BATTERY_NNEW` (**default 64 — set 2–8 on a rotating disk, or one arm costs
hours**), `BATTERY_TIMEOUT=0` (unlimited; `rc=124` in the JSON = this wall, not a fault),
`BATTERY_FRAC`, `BATTERY_MAXT`, `BATTERY_STREAM=1`, `BATTERY_WARM` (0 = cold, what the
window read-matrix wants), and the two added for the E19/E20 levers:

| Env | Adds | Why |
|---|---|---|
| `BATTERY_HOT_LAYERS=<file>` | `HOT_LAYERS` to every arm | the per-layer schedule vs one global `HOT_FRAC`, same runner, same bytes |
| `BATTERY_WIN_MB=<n>` | `Q36_WIN_MB` to the non-W arms | window on/off on the *same* arm set (the `W0/W256/W1G` arms set their own) |

```bash
# the window read-matrix, cold, on a packed container (n_new small: HDD)
BATTERY_ARMS="W0 W256 W1G" BATTERY_NNEW=8 BATTERY_TIMEOUT=0 BATTERY_FRAC=0.25 \
BATTERY_MAXT=8192 BATTERY_STREAM=1 tools/run_battery.sh ~/q27_i4-i4 ~/prompt_7k.txt
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
## 11. Dense-trunk int4 packer (`tools/pack_trunk_i4.py`)

Why: the field container's dense trunk is **F16** (54.7 GB) — the pinned converter
quantizes routed experts only and a dense model has none. Decode is pure weight
streaming, so this is the only structural lever on the s- per-token floor, and it
unblocks the window cache, the sorted container (arm E) and honest byte accounting.
Report: [COST_MODEL_QWEN36.md](COST_MODEL_QWEN36.md).

```bash
python3 tools/pack_trunk_i4.py <src_dir> --dry-run          # target set + byte budget
python3 tools/pack_trunk_i4.py <src_dir> <out_dir> --verify [--hash]
python3 tools/pack_trunk_i4.py <src_dir> <out_dir> \
        --profile ~/Models/q27_i4/hot_profile.json    # hot sorted prefix in the same pass
```

| Option | Meaning |
|---|---|
| `--components a,b,…` | subset of `lmhead,shexp,attn,dnproj,dnout,router` (default all). Attention + DeltaNet input projections cost the most perplexity per byte saved — keep them F16 if quality matters more than bytes |
| `--profile <hot_profile.json>` | permutes MLP `gate`/`up` rows and `down` columns by `argsort(-energy)` (hot_sort's rule) **before** quantizing, so the hot prefix is the file prefix — what `HOT_FRAC`'s contract requires. A profile from another geometry is a refusal, and a partial profile names the layers that would serve unsorted |
| `--block-rows N` | streaming row block (default 256 → ~2 MB RAM; runs on the 7.6 GB box) |
| `--emit-dequant <dir>` | second container whose targets are the **dequantized** values as F16 — the oracle a parity gate runs against, since the engine's mapped-F16 read of exactly what the int4 read reconstructs isolates layout bugs from every other numeric |
| `--verify` | re-reads the output and proves the engine's dispatch predicate (`dtype == U8`, `nbytes == I*O/2`, `F32 .qs` with `numel == O`) plus byte-identity of the passthrough tensors |
| `--hash` | sha256 every shard in and out (slow: a full re-read) |

No engine change is needed: the mapped branch dispatches on the dtype predicate
alone. Serve the output exactly as the F16 container (`TRUNK_MAP=1`, launcher
levels unchanged). Offline tests: `python3 tools/test_pack_trunk_i4.py`
(layout oracle, `pack∘sort == sort∘pack` byte-for-byte, predicates, refusals).

## 12. Cost model / projection (`tools/project_q36.py`)

Answers "what will this cost **on this host**" instead of quoting a fixed number.
Per-class bytes are computed from the container's own shard headers; every host
parameter is a named field anchor with provenance, and `--field-check` fails if the
model drifts out of band against the recorded measurements.

```bash
python3 tools/project_q36.py --container ~/Models/q27_i4 --frac 0.25 --ctx 1524
python3 tools/project_q36.py --geometry --field-check          # gate the model itself
python3 tools/project_q36.py --container ~/Models/q27_i4 --probe <big file>   # this host's rate
```

Read the output as: the attribution table (what the dial governs and what it does
not), TTFT, cold and warm s/token, the frac at which the working set fits RAM, and
the field fit. `--json <path>` writes everything for scripts.

| Option | Meaning |
|---|---|
| `<dir>` or `--container <dir>` | either form; the geometry and dtype mix are read from the container |
| `--win-mb N` | model the Phase-4b window cache (`Q36_WIN_MB`) using the engine's own capacity formula, incl. the 64-rows/layer floor |
| `--layer-fracs <json>` | a `HOT_LAYERS` file — the effective mean frac replaces the global one |
| `--stack` | the whole ladder (F16 → int4 → +schedule → +window → +speculation), every row model output |
| `--probe <file>` | replace both rates with this host's own measured sequential read rate |
| `--field-check` | re-derive the recorded field points and exit non-zero outside band |

## 13. Per-layer HOT_FRAC schedule (`tools/make_layer_fracs.py`)

`HOT_FRAC` is one scalar; layers compress differently. This spends a resident-byte
budget where it captures the most activation energy, and writes the file the engine's
`HOT_LAYERS` scan can actually read.

```bash
q36-minipc fracs <model_dir> [out]                   # -> <model_dir>/hot_layers.json
python3 tools/make_layer_fracs.py <model_dir> --budget-gb 5.5 --check -o hot_layers.json
```

| Option | Meaning |
|---|---|
| `--budget-gb F` | resident MLP bytes to spend (default: `--ram-gb` minus the fixed residents) |
| `--ram-gb F` | derive the budget from RAM (default 7.62, the field box) |
| `--trunk-dtype int4\|f16` | bytes per neuron (int4 = 0.5 B/element; the default) |
| `--min-frac`/`--max-frac` | per-layer clamps (the floor cannot be traded away and is reported if it overruns the budget) |
| `--normalize` | allocate by each layer's *relative* energy instead of absolute |
| `--check` | re-parse the output with a re-implementation of the engine's own scan |
| `--json-report <p>` | per-layer K, captured energy, bytes, and the gain vs a uniform frac |
| `--budget-sweep GB,GB,…` | write one schedule per budget (`<stem>-<gb>g.json`) + the bytes/energy row for each — the resident-budget curve the quality arms are run against |

Output format is a **flat** `{"L0": 0.55, "L1": 0.4, …}` map — the engine parses it
with `strstr("\"L")` + `atof`, so nothing else may live in that file. The launcher
picks up `<model_dir>/hot_layers.json` automatically (`Q36_HOT_LAYERS` overrides it).
Gates: `python3 tools/test_layer_fracs.py`.

## 14. The two measurement verdicts (`scripts/`)

Both take artifacts a run already produces and return a verdict, so a field session ends
with an answer rather than a JSON nobody reads. Unlike the launcher's tools these live in
the repo, so run them from the checkout (`~/bonsai/scripts/...`).

### 14.1 Window read-matrix — `scripts/check_window_readmatrix.py`

Phase 4b's gate 3, executable: window-on must cut read bytes ≥40% vs `W0` at equal-or-better
decode, and the gate **states its `n_new`** (E14's calibration lesson — at the field's
one-step budget a 40% *total* cut is arithmetically out of reach for any window).

```bash
BATTERY_ARMS="W0 W256 W1G" BATTERY_NNEW=8 BATTERY_TIMEOUT=0 BATTERY_FRAC=0.25 \
BATTERY_MAXT=8192 BATTERY_STREAM=1 \
tools/run_battery.sh ~/Models/q27_i4-i4 ~/prompt_7k.txt        # then:
python3 scripts/check_window_readmatrix.py results/battery_<host>_<ts>.json
```

Exit codes: `0` PASS (names the budget that clears the bar, and any that do not), `1` FAIL
(nothing cleared it — the knob stays default-off per the contract), `2` INCONCLUSIVE
(the `hits 0 misses 0` signature: the window never allocated, i.e. the trunk is not packed;
or `n_new < 8`). `--self-test` rehearses all four verdicts and is in CI.

### 14.2 MTP draft depth — `scripts/exp_mtp_depth_q36.py`

"How much does one more draft buy" is a sweep of an existing knob plus a parse of the engine's
own report line — no engine change, and the generative half (tree/multi-candidate drafts) is
**not** implemented:

```bash
python3 scripts/exp_mtp_depth_q36.py ~/Models/q27_i4-i4 ~/prompt.txt --depths 0,1,2,3 --nnew 64
```

Prints tokens/forward, acceptance and the effective tok/s per depth, and refuses to report
anything on a container without an MTP head (`meta.mtp`) instead of inventing numbers.
Real acceptance is a real-weights measurement — the field's M1 arm measured 82.0% /
2.86 tokens/forward.

### 14.2b The physical floor — `q36-minipc floor` / `project_q36.py --floor`

`project` prices a *configuration*; `floor` answers "how low can THIS container go on THIS
box, and what is in the way". It is computed from the container's own headers, and it exists
because `project`'s warm column credits only the MLP as cacheable (right for the window,
wrong as a bound): every class is read exactly once per decode token, so

    streamed_bytes_per_token = max(0, total_bytes(frac) - cache_room)
    cache_room               = RAM - (tensors that must stay resident: embed, KV, norms)

```bash
q36-minipc floor                        # your model, your RAM, with the lever table
Q36_EMBED_STREAMED=1 Q36_KVQ=kv8 q36-minipc floor   # with the RAM levers applied
python3 tools/project_q36.py <container> --floor --ram-gb 6.9 --embed-streamed --kvq kv8
```

It prints the lever table (bytes removed per lever and the s/token each buys), `frac_zero` —
the frac at which **nothing streams any more**, past which decode is compute-bound and the
model has no term for it — and the floor with every lever applied at once. `--embed-streamed`
models the one lever that needs engine work: the engine loads the embedding table f32-resident
(`load_t_n` → `falloc` + `st_read_f32`), which at 27B geometry is 5.09 GB of a 7.62 GB box
(2.54 GB f16) for a table decode touches one 20 KB row of.

### 14.2c `Q36_EMBED_MAP=1` — the vocabulary table stays on disk

`model.embed_tokens.weight` is `vocab*hidden` = 1.27e9 elements: **2.54 GB as f16, 5.09 GB as
f32**, which the engine loads resident (`load_t_n` → `falloc` + `st_read_f32`) to serve ONE
20 KB row per decode token. On a 7.62 GB mini-PC that is a third of RAM doing nothing, and
the page cache it frees holds exactly that much more of the per-token working set:
`q36-minipc floor` prices it as +2.54 GB of `cache_room` (alone: 265 → 38 s/token; with an
int4 lm_head: the streamed set goes to zero).

```bash
Q36_EMBED_MAP=1 q36-minipc bench ~/Models/q27_i4-i4
Q36_EMBED_MAP=1 q36-minipc serve ~/Models/q27_i4-i4
bash scripts/check_embed_map.sh          # the gate: id-identity + engagement
```

Rows are read through `st_read_slice_f32` (bounds-checked, f16/bf16→f32, over the tensor's own
persistent fd), so a row is one pread of `D*esz` bytes. A table that is not a float tensor
(int4/affine) is refused with a message and stays resident; the engine prints
`[embed] mapped: N row read(s)` at exit, which is the gate's proof that the path ran.

### 14.2d Low-rank + sparse residual (LRS) — `tools/analyze_lrs.py`

The LRS proposal replaces a dense GEMV with `W ~= U @ V.T + S` (rank r + the top-k residual
entries) for the classes with no HOT_FRAC dial. Its sketch prices the win against **F16**
("1.5-3x on the untouchable ~70%") — the wrong baseline, because the container it targets is
already **int4** at 0.5 B/element. This tool answers the only question that decides the lever,
with both sides *measured*:

```bash
# on the SOURCE container (before pack_trunk_i4 -- a packed U8 tensor is already int4,
# so quantizing it again would compare int4 with itself; the tool says so and skips it)
python3 tools/analyze_lrs.py ~/Models/q27_i4 --classes deltanet,lm_head,attn --rank-scan 8,32,128
python3 tools/analyze_lrs.py ~/Models/q27_i4 --rows 2048 --classes lm_head   # big tensors
python3 tools/analyze_lrs.py --self-test                                     # in CI
```

Per tensor it prints int4's rel-L2 at its byte cost (`m*n/2 + m*4`), then LRS at each rank with
the sparse budget being **whatever bytes are left**, so the comparison is byte-for-byte and
never flatters either side. `--self-test` builds a genuinely rank-16 matrix where LRS MUST win
and an iid Gaussian one where it cannot, and asserts both verdicts, exact byte accounting, error
monotonicity in the sparse budget, and that the int4 baseline reproduces this repo's measured
0.12-0.14 rel-L2 band on random weights.

Measured on the synthetic container (`--classes all`, ranks 8/32/128): **LRS loses on 5/5**,
needing 3.4-6.0x MORE bytes to reach int4's error, and getting *worse* as rank rises at fixed
bytes (a higher rank eats the sparse budget that was carrying the approximation). Random weights
are the worst case for low rank by construction, so the number that decides the lever is a
`--rows` run on real weights. See `docs/ROADMAP.md` E25.

### 14.3 Cascade-Hot Streaming (CHS) — `tools/train_chs_predictor.py`

CHS's runtime half is withheld pending a measurement (see `docs/CHS_QWEN36.md`); what runs
today is the offline half, against the dumps the engine already writes:

```bash
python3 tools/train_chs_predictor.py --dumps ~/dumps/train --D 5120 --I 17408 --K 64 \
        --delta-report                                  # the premise: 27.8%/token measured
python3 tools/train_chs_predictor.py --dumps ~/dumps/train --dumps-eval ~/dumps/eval \
        --D 5120 --I 17408 --K 64 --rank-scan 16,32,64,128 --rank 64 \
        -o MODEL/chs_predictor.npz                      # the pre-registered gate
python3 tools/test_chs_predictor.py                     # the gates (also run in CI)
```

`--dumps-eval` is what makes it the pre-registered disjoint-corpus gate; without it the run
is a same-corpus holdout and says so. The artifact records `gate_cleared`; a runtime must not
prefetch from a map whose gate did not clear. Dump records are `D f32 | K u32 | K f32` and
**must** be read with a structured dtype (a bare-f32 read makes every index 0).

---

*Everything here is synthetic-container measured; real-checkpoint validation
is COMPLETE for W1 dense (docs/REALWEIGHTS_QWEN36.md, tag `phase-5`). If a command
misbehaves, run `q36-minipc doctor` and read the stderr banner before anything else.*
