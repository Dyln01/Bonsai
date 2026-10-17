# Installing Bonsai — full guide (Linux & Windows)

Bonsai runs **full-size dense 27B-class and MoE Qwen3-family models on 8 GB
mini-PCs** — no GPU, no downsizing. It is a **patch stack on the open-source
[colibri](https://github.com/JustVugg/colibri) inference engine** (Apache-2.0),
not a fork: you clone colibri, overlay Bonsai's engine files, build, and use
the `q36-minipc` launcher to compose the small-RAM profile.

There are two things to install:

| Component | What it is | Where it comes from |
|---|---|---|
| **Engine + launcher** | `qwen36.c` patch stack (TRUNK_MAP, hot-neuron paging, GRIFFIN, KV8, KV_TQ, flash prefill) + `q36-minipc` | this repo + colibri upstream |
| **Model container** | a directory with `qwen36_meta.json`, `config.json`, `tokenizer.json`, safetensors shards | synthetic (zero download, what all our benchmarks use) **or** a real Qwen3 container via colibri's converter |

> **Honesty note:** every Bonsai performance number in the reports was
> measured on **synthetic same-geometry containers** (no real checkpoints were
> downloaded or used in testing). Real-container quality validation is the
> declared open step — see the README *Status* section.

Once installed, **[docs/COMMANDS.md](COMMANDS.md) lists every command, knob
and log line** — launcher, engine, wire protocol, generators, drivers, patch
scripts and verification.

---

## 1. Requirements

| | Minimum (mini-PC target) | Comfortable (dev box) |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores, x86-64 | 8+ cores |
| Disk | ~40 GB free (containers 0.4–23 GB) | NVMe recommended |
| GPU | none (CPU only) | none needed |
| OS | Linux x86-64, or Windows 10/11 via WSL2 | same |

Software: `git`, `python3` (≥3.8) with `numpy`, and `build-essential`
(gcc + make + OpenMP). That's the complete dependency list for the CPU path.

---

## 2. Linux — native install

### 2.1 Dependencies

```bash
# Ubuntu / Debian
sudo apt install git python3 build-essential python3-numpy
# Fedora: sudo dnf install git python3 gcc make python3-numpy
# Arch:   sudo pacman -S git python gcc make python-numpy
```

### 2.2 Get colibri + Bonsai

```bash
cd ~
git clone https://github.com/JustVugg/colibri
git clone https://github.com/Dyln01/Bonsai bonsai
```

### 2.3 Overlay the Bonsai engine files

```bash
cp ~/bonsai/engine/qwen36.c        ~/colibri/c/
cp ~/bonsai/engine/kv8_fused.h     ~/colibri/c/
cp ~/bonsai/engine/kv8_parity.c     ~/colibri/c/
cp ~/bonsai/engine/kv_tq.h         ~/colibri/c/
cp ~/bonsai/engine/kv_tq_parity_q36.c ~/colibri/c/
```

### 2.4 Build

```bash
make -C ~/colibri/c qwen36
```

This produces `~/colibri/c/qwen36` (the engine binary). First build takes a
few minutes — `qwen36.c` is a large single translation unit.

### 2.5 Verify the install (parity gates — do this once)

The KV codecs carry their own test programs; both must pass before you trust
any run:

```bash
gcc -O2 -I"$HOME/colibri/c" -o /tmp/kvtq "$HOME/colibri/c/kv_tq_parity_q36.c" -lm && /tmp/kvtq
gcc -O2 -I"$HOME/colibri/c" -o /tmp/kv8p  "$HOME/colibri/c/kv8_parity.c"        -lm && /tmp/kv8p
```

Expected last lines:

```
[info] n=128 mean rel L2 err: rot-int4=0.096 polar4=0.147 polar3=0.291
PARITY OK: FWHT, radius, direction, linearity, inert, accounting — all gates passed
...
```

### 2.6 Install the launcher

```bash
cp ~/bonsai/tools/q36-minipc.sh ~/bin/q36-minipc
cp ~/bonsai/tools/hot_sort.py    ~/bin/hot_sort.py
chmod +x ~/bin/q36-minipc ~/bin/hot_sort.py
# make sure ~/bin is on PATH (usually already is; else add to ~/.profile):
#   export PATH="$HOME/bin:$PATH"
```

The launcher expects the engine at `~/colibri/c/qwen36` (override with
`Q36_ENGINE=<path>`).

### 2.7 Get a model container

#### Option A — synthetic container (zero download, recommended first)

Same geometry as the real models, random weights — exactly what all Bonsai
benchmarks run on. No network needed, safe license-wise, minutes to generate.

```bash
# small dense student (~1–2 GB on disk) — the fastest smoke test
python3 ~/bonsai/scripts/gen_student_i4.py

# full-shape Qwen3-Coder-30B-A3B MoE geometry (48L × 128 experts, int4)
python3 ~/bonsai/scripts/gen_coder30_synth.py

# full 27B dense shape in f16 (large, ~49 GB — for TRUNK_MAP experiments)
python3 ~/bonsai/scripts/gen_qwen38_27b_f16.py
```

Generators write to `~/synth_models/<name>` and need only `numpy`. Add a
minimal tokenizer where one is missing:

```bash
python3 ~/bonsai/scripts/make_tokenizer.py ~/synth_models/<name>/tokenizer.json <eos_id>
```

#### Option B — real Qwen3 container (your download, your hardware)

colibri upstream maintains the real-model path. Either pull a pre-converted
container:

```bash
hf download Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64 --local-dir ~/Models/qwen36_i4_gs64
# (~23 GB, self-contained int4 container)
```

or convert the original checkpoint yourself (~70 GB bf16 download):

```bash
python3 ~/colibri/c/tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B \
       --out ~/Models/qwen36_i4_gs64 --gs 64
```

See colibri's `docs/qwen36.md` for the full model list (architecture-identical
checkpoints run unchanged).

### 2.8 First run

```bash
q36-minipc doctor ~/synth_models/student_1b_i4      # knob/model/level report
q36-minipc serve  ~/synth_models/synth_coder30     # MoE: OpenAI-compatible server
q36-minipc chat   ~/synth_models/student_1b_i4     # interactive chat
q36-minipc bench  ~/synth_models/student_1b_i4     # honest phase-split battery
```

On startup the engine prints its resolved config to stderr — **read the
banner before trusting a run**:

```
== qwen36 Phase-2 engine | cache=48/layer bits=4 ctx=8192 pilot=2 wide=1 ... ==
[qwen36] KV_TQ=4 codec=1 (rot-int4): KV rows packed 64 B + f32 radius/row ...
```

---

## 3. Windows

### 3.1 Path A — WSL2 (recommended; this is where every Bonsai number was measured)

All Linux steps run unchanged inside WSL2, with three measured rules that
matter on Windows:

**Install WSL2 + Ubuntu** (admin PowerShell):

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, launch Ubuntu, create your Linux user, then follow the
**Linux guide above verbatim** (§2.1–2.8) *inside* the distro.

**Rule 1 — work in the Linux filesystem, never `/mnt/c`.**
Clone repos, build, and keep model containers under `~/…` (ext4), not under
`/mnt/c/…` (drvfs). Measured on this stack: `O_DIRECT` does not work on the
Windows mount, and cold random reads run ~50–100× slower than on a real NVMe
ext4 volume. Access files from Windows via `\\wsl$\Ubuntu\home\…` in Explorer.

**Rule 2 — keep the WSL disk (and containers) off C: if you're short on space.**
WSL's virtual disk lives on `C:` by default. Move the whole distro before
generating big containers:

```powershell
wsl --shutdown
wsl --export Ubuntu D:\wsl\export\ubuntu.tar
wsl --unregister Ubuntu
wsl --import Ubuntu D:\wsl\ubuntu D:\wsl\export\ubuntu.tar --version 2
```

(The default user set in `/etc/wsl.conf` survives the move.)

**Rule 3 — never `wsl --shutdown` while a generation or benchmark runs.**
Killing the VM mid-write can push ext4 into read-only recovery mode. A killed
container generator leaves 0-byte shard tails — repair by writing only the
missing pieces, don't regenerate from scratch.

**Optional — cap WSL memory on small machines.** On an 8 GB PC, give WSL most
but not all RAM via `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=6GB
swap=8GB
```

(this also makes `q36-minipc`'s auto RAM detection behave like a real mini-PC,
which is exactly the hardware this stack targets).

### 3.2 Path B — native Windows (possible, untested for this stack)

Upstream colibri fully supports native Windows: prebuilt release archives,
`START-HERE.bat` one-step setup, and MSYS2/MinGW builds — see colibri's
`docs/windows.md`. To run the Bonsai patch stack natively, overlay the engine
files exactly as in §2.3 into a Windows colibri checkout and build with
`make qwen36` in the MSYS2 shell.

Caveat, stated plainly: **the Bonsai patch stack is built and validated on
Linux/WSL2 only.** The OS-sensitive parts are guarded (`madvise` sits behind
`#ifdef MADV_WILLNEED`, upstream ships Windows compat headers), so it should
compile — but no Bonsai number on this repo was measured on native Windows.
If you try it, run the §2.5 parity gates first; they are pure C99 and will
confirm the codecs ported cleanly.

---

## 4. Day-one profiles on a small box

`q36-minipc` composes the profile automatically (it reads
`qwen36_meta.json` to detect MoE vs dense). The knobs:

| Knob | Values | Meaning |
|---|---|---|
| `Q36_LEVEL` | `eco` / `balanced` / `quality` | preset: expert cap, threads, ctx, hot fraction |
| `Q36_KVQ` | `f32` / `kv8` (default) / `tq4` / `tq3` | KV cache tier (see below) |
| `Q36_HOT_FRAC` | 0.25 … 1.0 | dense only: fraction of MLP neurons computed (1.0 = exact) |
| `Q36_GRIFFIN` | 0/1 | dense only: per-prompt neuron selection (default 1) |
| `Q36_RAM_GB` | auto | tells the launcher the real RAM; caps the expert cache accordingly |

**KV tiers** (lossless → smallest): `f32` exact · `kv8` fp8 rows (−16/−26% KV
RAM) · `tq4` rotated-int4 rows, half of fp8, ~0.10 rel L2 error · `tq3`
PolarQuant-3 rows (~0.31 B/row at head_dim 128 — the RAM extreme).

**MoE model** (e.g. Qwen3-Coder-30B-A3B class):

```bash
q36-minipc seed  ~/Models/qwen36_i4_gs64   # once: Zipf .coli_usage prior
q36-minipc serve ~/Models/qwen36_i4_gs64   # cap + PIN + PILOT + WIDE + KV tier
```

Measured on the synthetic full-shape 30B (4-core/7.8 GB emulation): balanced
profile ≈ 7.3 GB RSS, 96–99.6% expert cache hit, 3.4–8.2 tok/s decode
depending on workload locality, TTFT 17–20 s at ~350-token prompts.

**Dense model** (27B-class): stock loads OOM on 8 GB (measured, rc=137). The
Bonsai path is mmap + neuron paging:

```bash
q36-minipc calibrate ~/Models/qwen27_i4   # once: Q36_CORPUS=<your_text.txt>
q36-minipc sort      ~/Models/qwen27_i4   # once: hotness-sorted copy
q36-minipc serve     ~/Models/qwen27_i4-sorted
# or skip both: GRIFFIN adapts per-prompt on the UNMODIFIED container
```

Measured on the sorted student analog: MLP phase 119→38 ms/token (3.1×),
decode 5.8→11.0 tok/s, TTFT 26.5→11.6 s, RSS ≈ 1.9–2 GB; the full-size 27B
projects to a ~4 GB envelope at frac 0.25 (12–14 tok/s warm) vs OOM stock.

---

## 5. Verification checklist

After install, all of these should hold:

1. `make -C ~/colibri/c qwen36` completes with no errors.
2. `kv_tq_parity_q36` and `kv8_parity` both print `PARITY OK`.
3. `q36-minipc doctor <model>` lists knobs with nonzero ref counts
   (`TRUNK_MAP`, `GRIFFIN`, `KV8`, `KV_TQ`, `Q36_FLASH`, …).
4. A first `serve`/`chat` run shows the engine banner with your resolved
   level/ctx/KV tier, and finishes with `rc=0` (no crash, finite output).
5. `q36-minipc bench <model>` prints `[timers] decode: N tokens` — report
   `TOTAL ms/token` as decode speed. (Ignore the engine's `Speed:` line:
   it divides by total wall time including prefill.)

---

## 6. Troubleshooting (all measured on this stack)

| Symptom | Cause | Fix |
|---|---|---|
| Dense model killed with rc=137 (OOM) at load | stock loader tries to keep ~22 GB resident | run via `q36-minipc` (sets `TRUNK_MAP`); never load dense 27B-class stock on 8 GB |
| Decode crawls, expert hit-rate collapses on a small box | expert cache grew past RAM and thrashes | `Q36_LEVEL=eco` or `Q36_RAM_GB=<real RAM>`; the launcher sets `CAP_RAISE=0` for you — don't hand-set a big `CAP` on an 8 GB box |
| Everything 50× slower than expected on Windows | files live under `/mnt/c` (drvfs) | move engine + containers into the ext4 home (`~`) — §3.1 Rule 1 |
| A generated container has 0-byte shards | `wsl --shutdown` or reboot hit mid-generation | write only the missing layer shards/config, don't regenerate the whole container |
| `GEN`/`NGEN` env seems ignored | that variable is dead in this engine | use `N_NEW=<n>` for generated-token count |
| Prompt length surprises you | the tokenizer is byte-level: 1 token ≈ 1 byte, not 1 word | read the engine's `[enc] prompt tokens: N` line, never trust the filename |
| Flash prefill not engaged (TTFT ~2× expected) | flash is opt-in | `Q36_FLASH=1` (or use the launcher's bench, which reports both) |
| Two runs give different tokens | unpatched-path nondeterminism (timing-sensitive speculation gate) on synthetic fixtures | expected; gate correctness with the parity programs + codec error bounds, not token hashes |

---

*Engine provenance: `engine/qwen36.c` is a snapshot of colibri's `c/qwen36.c`
@ bf24429 plus the Bonsai patch stack; `engine/NOTICE.md` states attribution.
Bonsai does not include or fork colibri — clone it separately as above.*
