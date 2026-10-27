# Benchmarking a dense 27B (Qwen3.8-27B class)

The dense 27B is the model the whole Bonsai stack exists for: stock, it needs
~20 GB of resident weights and is **OOM-killed on an 8 GB box (measured,
rc=137)**. The Bonsai path (mmap'd trunk + neuron paging + GRIFFIN + tiered
KV + flash prefill) runs the same full-size model inside a ~4 GB envelope.

This page is the complete recipe: install → container → benchmark matrix →
how to read the numbers → day-2 serving. Sizes below are measured, not
estimated.

---

## 0. Pick your container

| Container | Size on disk | Download | What it tells you |
|---|---|---|---|
| **A. student_1b_i4** (synthetic) | 1.4 GB | none | the full dense machinery in minutes — same kernels, small geometry. Start here, always. |
| **B. synth_qwen38_27b_f16** (synthetic, full 27B shape) | 46 GB | none | exact Qwen3.8-27B geometry (64L: 16 GQA + 48 DeltaNet, hidden 5120, MLP 17408, vocab 248320), random weights. RAM envelope + speed of the real shape. |
| **C. real Qwen/Qwen3.8-27B** (converted) | 56 GB bf16 → **~50 GB F16** (the pinned converter quantizes routed experts only; a dense 27B has none, so its MLP stays F16 — the int4-trunk figures elsewhere are the engine's load/map-time dialect, not this on-disk container) | yes (~56 GB) | the true end-to-end number incl. weight quality. Everything below applies unchanged. |

Quality note: A and B have random weights — they measure the **machinery**
(RAM, TTFT, ms/token) exactly, but not model quality. Every published Bonsai
number is container-A/B-measured; container-C is your validation.

---

## 1. Install (5 commands + 2 checks)

```bash
sudo apt install git python3 build-essential python3-numpy
git clone https://github.com/JustVugg/colibri   # engine (Apache-2.0)
git clone https://github.com/Dyln01/Bonsai bonsai
cp ~/bonsai/engine/qwen36.c ~/bonsai/engine/kv8_fused.h ~/bonsai/engine/kv8_parity.c \
   ~/bonsai/engine/kv_tq.h ~/bonsai/engine/kv_tq_parity_q36.c ~/colibri/c/
make -C ~/colibri/c qwen36
cp ~/bonsai/tools/q36-minipc.sh ~/bin/q36-minipc && cp ~/bonsai/tools/hot_sort.py ~/bin/
chmod +x ~/bin/q36-minipc ~/bin/hot_sort.py
```

Verify (both must print `PARITY OK`/`PASS` — see [COMMANDS.md §9](COMMANDS.md)):

```bash
gcc -O2 -I"$HOME/colibri/c" -o /tmp/kvtq "$HOME/colibri/c/kv_tq_parity_q36.c" -lm && /tmp/kvtq
gcc -O2 -I"$HOME/colibri/c" -o /tmp/kv8p "$HOME/colibri/c/kv8_parity.c" -lm && /tmp/kv8p
```

Windows: do all of this inside WSL2 Ubuntu — the three WSL rules are in
[INSTALL.md §3.1](INSTALL.md) (work in the ext4 home, never `/mnt/c`;
containers go where the disk is).

---

## 2. Generate the containers (no download)

```bash
# A. the 5-minute smoke test
python3 ~/bonsai/scripts/gen_student_i4.py
q36-minipc doctor ~/synth_models/student_1b_i4
q36-minipc bench   ~/synth_models/student_1b_i4        # first real numbers

# B. the full 27B shape (~46 GB — check free disk first)
python3 ~/bonsai/scripts/gen_qwen38_27b_f16.py         # takes a while; never wsl --shutdown mid-write
```

Container C (real model) instead of B:

```bash
# convert a local HF download of Qwen/Qwen3.8-27B (56 GB) to a ~50 GB F16 container
# (budget the disk: the converter quantizes ROUTED experts only — a dense 27B has none,
#  so its MLP stays F16 on disk; the engine's int4 trunk is a load/map-time dialect):
python3 ~/colibri/c/tools/convert_qwen36.py --model <Qwen3.8-27B download> --out ~/Models/q27_i4
# (the converter also takes --repo Qwen/Qwen3.8-27B to download in-stream;
#  --ebits/--gs tune expert quantization — dense 27B uses none of that, the MLP is the "shared expert")
```

---

## 3. The benchmark matrix

`q36-minipc bench <model>` runs the honest battery: phase-split timers,
peak RSS, TTFT. Dial the knobs per arm; change ONE thing per arm.

```bash
M=~/synth_models/synth_qwen38_27b_f16     # or ~/Models/q27_i4 for the real one

# --- arm 1: the stock attempt (proves the OOM, sets the baseline) ---
q36-minipc doctor $M
SNAP=$M Q36_MAXT=2048 OMP_NUM_THREADS=4 ~/colibri/c/qwen36 32 4 ~/some_prompt.txt
#   on an 8 GB box: killed, rc=137. On a 16 GB box: boots but slow (full 46 GB f16 sweep/token).
#   E11 (2026-10-23): this arm is EXACTLY what tools/run_battery.sh launched on the
#   mini-PC until now — no TRUNK_MAP=1 and the inert Q36_HOT_FRAC alias. The field JSONs
#   results/battery_llm-030_* (rc=137, identical ~5.1 GB fs-read watermark) are arm-1
#   re-proofs, NOT Bonsai-profile numbers. The runner now ships the dense dialect,
#   records the operating point in its JSON, and scripts/check_runner_env.py guards the
#   env names in CI. Re-run: BATTERY_FRAC=0.25 BATTERY_MAXT=8192 tools/run_battery.sh ...
#   E12 (2026-10-24): the re-run PROVED that fix (no OOM; peak RSS ~7.0/7.62 GB; no
#   signal-9) and then died at the runner's old hardcoded `timeout 3600` — one cold
#   prefill pass at frac 0.25 (~41 GB) takes ~60-90 min at the field-measured
#   10-12 MB/s, so no arm finished. BATTERY_TIMEOUT is now a knob (0 = unlimited);
#   on a rotating disk probe ONE arm first:
#   BATTERY_ARMS="A" BATTERY_NNEW=2 BATTERY_TIMEOUT=0 BATTERY_FRAC=0.25 BATTERY_STREAM=1 ...
#   E13 (2026-10-25): the probe LANDED — first completed field arm (rc=0, wall 1:40:06,
#   zero swaps, peak RSS 6.94/7.62 GB). 1524-token prompt → TTFT 3676.22 s; decode
#   2,225,442 ms/token (one decode step; MLP trunk streaming = 71.7% of it, attention
#   core 17.2 s — compute cores are <1%). fs_read 71.72 GB ≈ load touch + one cold
#   prefill pass + ~17-24 GB decode re-read. Measured cost model (fits the wall exactly):
#   T(arm) ≈ 105 s + 3676 s + (n_new−1) × 2225 s → 1.7 h @ n=2, 10.4 h @ n=16.
#   Numbers: results/battery_llm-030_20261025-101318.json (+ _logs/A.err phase splits);
#   analysis: ROADMAP E13. Remaining matrix ≈ one 10-h overnight run at n_new=2
#   (append A for a same-night drift control, +1.7 h):
#   BATTERY_ARMS="B C D W0 W256 W1G" BATTERY_NNEW=2 BATTERY_TIMEOUT=0 \
#   BATTERY_FRAC=0.25 BATTERY_MAXT=8192 BATTERY_STREAM=1 tools/run_battery.sh ...
#   Expectation, honest: the trunk stream is arm-invariant and the KV/window arms move
#   ≤ ~2.4 min of a 37-min token at this context — this matrix measures mechanics and
#   the W read-matrix, not speed wins.
#   E14 (2026-10-25): the overnight matrix LANDED — 6/6 arms rc=0 (B/C/D + W0/W256/W1G,
#   n_new=2), 9 h 41 m unsupervised vs the cost model's 10 h 01 m (−3.3%: the model is
#   validated and usable for phase-5 budgeting). Zero swaps, peak RSS 6.94–6.96/7.62 GB.
#   Decode 2234.9–2358.0 s/tok, TTFT 3287.6–3660.9 s: with the probe, all seven arms sit
#   inside a ±6% HDD noise band — no arm separation, exactly as pre-registered. fs_read
#   71.76–71.93 GB on EVERY arm (0.3% spread): KV codecs and GRIFFIN/window move no
#   bytes at 1524 ctx; decode is pure trunk streaming. 4b-slice-3 gate-3 window verdict:
#   FAIL, documented negative — W256/W1G cut reads 0.047%/0.018% vs the ≥50% bar, and
#   the window never allocated at all (int4-only guard vs the F16 trunk; `hits 0 misses 0`
#   is that signature). Q36_WIN_MB stays 0. All seven arms emit the same first greedy
#   token (lossy-KV argmax parity; quality deferred to hot_sort + phase-5).
#   Numbers: results/battery_llm-030_20261025-133007.json (+ _logs/, incl. the
#   [qwen36 win] lines); analysis: ROADMAP E14, WINDOWING doc §Gates. The field battery
#   is COMPLETE; next field step = phase 5 (PHASE5_FORCE=1 tools/run_phase5.sh ...).
#
#   E17 (2026-10-27): the first completed phase-5 tf arm (frac 1.0) returned ppl ==
#   vocab_size EXACTLY and exposed the reason the arms above all emitted the same
#   greedy token: map_f16_one (MAPFAST v3.1) lacked the F16->F32 exponent-bias
#   rebase, so EVERY mapped F16 weight in EVERY field run read 2^-112 of its true
#   value — the trunk never participated in the numerics. The 'чисwik' garbage and
#   the cross-arm output identity were that, not unsorted pruning or KV-argmax
#   parity. What survives from arms 1/E13/E14: every TIMING and IO number (bytes
#   streamed and FLOPs issued are unchanged by a conversion bug) — load 105 s,
#   TTFT ~3300-3700 s, decode ~2225-2358 s/tok, fs_read ~71.8 GB/arm, RSS
#   envelope, the cost model (−3.3%). What is void: every quality/logit-derived
#   reading. Fixed via tools/patch_mapf16_bias.py + the E17 CI gate (mapped-vs-
#   resident tf-NLL parity on a real-headed F16-dense container) + a runner
#   tripwire (ppl == vocab_size aborts the session). Analysis: ROADMAP E17.

# --- arm 2: the Bonsai default (mapped trunk + GRIFFIN + frac 0.25 + kv8) ---
q36-minipc bench $M

# --- arm 3: the exactness dial ladder (quality vs speed) ---
Q36_HOT_FRAC=1.0  q36-minipc bench $M     # exact full compute — the honest ceiling
Q36_HOT_FRAC=0.5  q36-minipc bench $M
Q36_HOT_FRAC=0.25 q36-minipc bench $M     # the mini-PC daily point

# --- arm 4: GRIFFIN on vs off (per-prompt selection vs static prefix) ---
Q36_GRIFFIN=0     q36-minipc bench $M     # static hot prefix only
Q36_GRIFFIN=1     q36-minipc bench $M     # default: adapts to THIS prompt, unmodified container

# --- arm 5: the KV tier ladder (long-context RAM) ---
Q36_KVQ=f32  q36-minipc bench $M
Q36_KVQ=kv8  q36-minipc bench $M          # default
Q36_KVQ=tq4  q36-minipc bench $M          # −19% total RSS vs f32 at 8k; decode-price in docs/KVTQ_QWEN36.md

# --- arm 6: levels (eco/balanced/quality presets) ---
Q36_LEVEL=eco      q36-minipc bench $M
Q36_LEVEL=quality  q36-minipc bench $M
```

Report BOTH cold and warm numbers on the mapped runs (first run pays the
page-cache fill; warm decode is the steady-state claim — see
[COMMANDS.md §4](COMMANDS.md) for the log lines to read).

---

## 4. Reading the output

```
== qwen36 Phase-2 engine | cache=… ctx=… ==     ← resolved config, read first
[trunk] mapped … [dense-i8] …                   ← trunk + dense quant engaged
[griffin] k=… selected …                        ← per-prompt neuron set
[warm] madvise(WILLNEED) over … GB              ← one-time page-cache warm
TTFT: 14.8 s                                    ← prefill cost
[timers] decode: N tokens … TOTAL 38.0 ms/token ← decode rate = 1000/38 = 26.3 tok/s
PEAK RSS: 2.01 GB                               ← the envelope number
Speed: … tok/s                                  ← total wall incl. prefill — NOT decode
```

## 5. What to expect (all from the reports; labeled)

| Measurement | Value | Status |
|---|---|---|
| stock dense 27B on 8 GB | OOM at 7.36 GB, rc=137 | **measured** (synthetic full shape) |
| mapped trunk boot | 7.11 GB RSS | **measured** |
| student analog, frac 1.0 / 0.5 / 0.25 | decode 5.8 / 7.3 / **11.0 tok/s**, TTFT 24.5 / 14.8 / 11.6 s, MLP phase 119 / 79 / 38 ms/tok, RSS ~2.0 GB flat | **measured** (same kernels) |
| full 27B @ frac 0.25 | ~4 GB envelope, **12–14 tok/s warm** (RAM-bound) vs 0.5–0.9 unpaged | **projection** from measured ratios |
| real Qwen3.8-27B int4 (upstream, no Bonsai paging) | 3.45 tok/s, 16-thread CPU | upstream's own measurement |
| KV tiers at 8k ctx (small container) | f32 46.9 → kv8 26.6 → tq4 5.3 tok/s; RSS 805.6 → 673.3 → 649.9 MB | **measured** — KV tiers are RAM levers, not speed levers |

If your numbers are wildly off: `q36-minipc doctor` first, then the
troubleshooting table in [INSTALL.md §6](INSTALL.md) (drvfs, cache-cap
thrash, prompt-token counts).

## 6. Day 2: serve it

```bash
# once: calibrate on YOUR corpus, sort a copy (or skip both — GRIFFIN adapts per-prompt)
#   what goes in the corpus? representative traffic — see the subsection below.
#   repo example you can start from: docs/examples/q36_corpus_example.txt
Q36_CORPUS=~/corpus.txt q36-minipc calibrate $M
q36-minipc sort $M
q36-minipc serve $M-sorted              # OpenAI-compatible; Q36_SLOTS=<conversations>
```

Agent traffic: keep the three prefix-reuse rules in
[COMMANDS.md §5](COMMANDS.md) — byte-exact append-only resends, one slot
per conversation, `Q36_MAXT` ≥ full transcript.

#### The calibration corpus (`Q36_CORPUS`), concretely

`calibrate` measures per-neuron activation energy over this text to decide the
hot-neuron ordering — so the file should read like your real traffic, not like
an essay. Rules of thumb: representative > long; diverse > repeated; put the
best samples first (the launcher reads the first `Q36_CAL_TOKENS`×~4 bytes,
default ≈4 KB); plain UTF-8 text; 20–100 KB total is plenty. Nothing is
generated from it and nothing leaves the machine.

A working example (also committed at
[`docs/examples/q36_corpus_example.txt`](examples/q36_corpus_example.txt) —
copy it and swap in your own logs):

```text
Write a function that reverses a string in Python.
The server returned HTTP 429 for 6 of 80 requests; summarize what likely caused it
and propose a retry policy with backoff.
{"tool": "shell", "command": "du -sh /var/log/*", "purpose": "find the largest log before rotating"}
Refactor this loop into a comprehension without changing behavior: for x in items: ...
Read the traceback and name the most likely root cause in one sentence: KeyError: 'user_id'.
Draft a polite follow-up email asking for review on a pull request open for four days.
```

Mix your actual domains in the same file — code for a coding box, tool-call
JSON for an agent box, prose for a chat box. Avoid one repeated line (it
biases the ranking) and avoid eval prose like wikitext (that is Phase 5's
corpus, different job). And remember: calibration is optional — with
`GRIFFIN=1` (default) the unmodified container adapts per prompt; the sorted
copy only makes every later run a bit cheaper.
