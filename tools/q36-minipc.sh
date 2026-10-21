#!/bin/bash
# q36-minipc v2 - one-knob launcher for qwen36 models (MoE AND dense) on small boxes.
#
# MoE containers  : cap + PIN + PILOT + WIDE + KV8 (web-ops composition)
# Dense containers: TRUNK_MAP + (GRIFFIN | hot-sort prefix) + HOT_FRAC + TRUNK_WARM + KV8
#   auto-detected from qwen36_meta.json num_experts.
#
# Commands: serve|chat|bench|seed (MoE) + calibrate|sort (dense pipeline) + doctor
set -euo pipefail

LEVEL="${Q36_LEVEL:-balanced}"
CMD="${1:-help}"
MODEL="${2:-}"
ARG3="${3:-}"
ENGINE="${Q36_ENGINE:-$HOME/colibri/c/qwen36}"
TOOLS="$HOME/bin"

die() { echo "q36-minipc: $*" >&2; exit 2; }

# render the engine's "[prefill] p=.." lines as a one-line progress bar on the
# terminal; every other line passes through on its own line (portable awk)
q36_render() {
  awk '{
    if ($0 ~ /^\[prefill\] p=/) {
      p=$2; sub(/^p=/,"",p); t=$4; sub(/^t=/,"",t); e=$5; sub(/^eta=/,"",e);
      n=int(p/5); bar="";
      for (i=0;i<20;i++) bar = bar (i<n ? "#" : ".");
      printf "\r[calib |%s| %3d%%  t=%ss eta=%ss   ", bar, p, t, e; np++
    } else { printf "\n%s", $0 }
  } END { printf "\n";
        if (np == 0) print "[minipc] note: no [prefill] progress lines from the engine — either the prompt was <=1 token, or the binary predates the progress patch (rebuild: cp engine/*.c engine/*.h into colibri c/ + make -C c qwen36)" > "/dev/stderr" }'
}

# model check only for commands that need one
case "$CMD" in
  help) META=""; NEXP=0; CLASS="none" ;;
  *)
    [ -n "$MODEL" ] && [ -f "$MODEL/qwen36_meta.json" ] || die "model dir must contain qwen36_meta.json"
    META="$MODEL/qwen36_meta.json"
    NEXP=$(python3 -c "import json;print(json.load(open('$META')).get('num_experts',0))" 2>/dev/null || echo 0)
    if [ "$NEXP" -gt 0 ]; then CLASS="moe"; else CLASS="dense"; fi
    ;;
esac

# ---- levels ----
case "$LEVEL" in
  eco)      CAP=24; THREADS=2; CTX=4096;  KV8V=1; WIDE=2; FRAC=0.25 ;;
  balanced) CAP=48; THREADS=4; CTX=8192;  KV8V=1; WIDE=1; FRAC=0.25 ;;
  quality)  CAP=64; THREADS=4; CTX=16384; KV8V=0; WIDE=1; FRAC=0.5  ;;
  *) die "unknown Q36_LEVEL '$LEVEL' (eco|balanced|quality)" ;;
esac

# ---- KV tier (Q36_KVQ): f32 | kv8 (default) | tq4 | tq3 ----
#   tq4 = KV_TQ=4 codec 1 (rotated int4, n/2 B/row, ~0.10 rel L2 — 2x under KV8)
#   tq3 = KV_TQ=3 codec 0 (PolarQuant-3, ~0.31 B/row @ HD=128 — the RAM extreme)
KVQ="${Q36_KVQ:-kv8}"
case "$KVQ" in
  f32)  KVENV="KV8=0" ;;
  kv8)  KVENV="KV8=$KV8V" ;;
  tq4)  KVENV="KV_TQ=4 KV_TQ_CODEC=1" ;;
  tq3)  KVENV="KV_TQ=3 KV_TQ_CODEC=0" ;;
  *) die "unknown Q36_KVQ '$KVQ' (f32|kv8|tq4|tq3)" ;;
esac

RAM_GB="${Q36_RAM_GB:-$(awk '/MemTotal/ {printf "%d", $2/1048576+0.5}' /proc/meminfo 2>/dev/null || echo 8)}"
if [ "$RAM_GB" -lt 8 ] && [ "$CAP" -gt 24 ]; then CAP=24; echo "[minipc] RAM<8GB: cap capped to 24"; fi

# ---- compose the profile ----
if [ "$CLASS" = moe ]; then
  COMMON="SNAP=$MODEL Q36_MAXT=$CTX OMP_NUM_THREADS=$THREADS COLI_NO_OMP_TUNE=1 CAP_RAISE=0 PIN=auto WIDE=$WIDE PILOT=2 SMOOTH=0.30 CONF_LIMIT=0.92 USAGE_SAVE=1 KVSAVE=1 $KVENV RAM_GB=$RAM_GB"
else
  # dense: trunk on disk; GRIFFIN by default (works on unmodified containers);
  # TRUNK_WARM fills the page cache once. HOT_LAYERS/FRAC still honored if user sets them.
  FRAC="${Q36_HOT_FRAC:-$FRAC}"
  COMMON="SNAP=$MODEL Q36_MAXT=$CTX OMP_NUM_THREADS=$THREADS COLI_NO_OMP_TUNE=1 TRUNK_MAP=1 GRIFFIN=${Q36_GRIFFIN:-1} HOT_FRAC=$FRAC TRUNK_WARM=1 $KVENV RAM_GB=$RAM_GB"
  [ -n "${Q36_HOT_LAYERS:-}" ] && COMMON="$COMMON HOT_LAYERS=$Q36_HOT_LAYERS"
fi

case "$CMD" in
  serve)  exec env $COMMON SERVE=1 OPENAI=1 ${Q36_SLOTS:+KV_SLOTS=$Q36_SLOTS} "$ENGINE" "$CAP" 4 ;;
  chat)   exec env $COMMON N_NEW=512 COLI_TEMP=0.7 "$ENGINE" "$CAP" 4 /dev/stdin ;;
  bench)
    PF="${ARG3:-/tmp/q36bench_prompt.txt}"
    [ -f "$PF" ] || printf 'def fibonacci(n):\n    if n < 2: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nWrite a function to check if a string is a palindrome. ' > "$PF"
    echo "[minipc] model class: $CLASS (num_experts=$NEXP)"
    env $COMMON N_NEW=64 COLI_TEMP=0 COLI_TIMERS=1 /usr/bin/time -v \
        "$ENGINE" "$CAP" 4 "$PF" > /tmp/bench.out 2> >(tee /tmp/bench.err | q36_render >&2)
    wait 2>/dev/null || true
    grep -E "== qwen36|RSS after load|hit rate|TTFT|decode: [0-9]+ tokens|TOTAL|PEAK RSS|Speed:|PIN|USAGE|warm|hot]" /tmp/bench.err || true
    echo "---"
    echo "decode tok/s = 1000 / (TOTAL ms/token); Speed: includes prefill."
    ;;
  seed)
    if [ "$CLASS" != moe ]; then echo "[minipc] seed is a MoE-cache prior; dense models self-adapt via GRIFFIN."; exit 0; fi
    if [ ! -f "$MODEL/.coli_usage" ]; then
      python3 - "$MODEL" <<'PYEOF'
import sys, json
import numpy as np
model = sys.argv[1]
cfg = json.load(open(f"{model}/qwen36_meta.json"))
L, E = cfg.get("n_layers", 48), cfg.get("num_experts", 128)
rng = np.random.default_rng(11)
lines = [f"-1 {L} {E}", "-2 1 qwen36"]
for li in range(L):
    ranks = np.arange(1, E+1); w = 1.0/ranks; w = w/w.sum()
    counts = np.rint(w*50000).astype(int)
    for e, c in zip(rng.permutation(E), counts):
        if c > 0: lines.append(f"{li} {e} {c}")
open(f"{model}/.coli_usage","w").write("\n".join(lines))
print(f"[seed] Zipf prior written: {L} x {E}")
PYEOF
    else echo "[seed] .coli_usage exists - keeping it."; fi
    ;;
  calibrate)
    # dense: run the engine over a corpus prompt, write the neuron profile
    [ "$CLASS" = dense ] || die "calibrate is for dense containers"
    OUT="${ARG3:-$MODEL/hot_profile.json}"
    CORPUS="${Q36_CORPUS:-$MODEL/calib_prompt.txt}"
    [ -f "$CORPUS" ] || die "set Q36_CORPUS=<text file> for calibration"
    # Bound the run by default: full-compute prefill is ~1-2 s/token on dense 27B
    # at mini-PC core counts. Q36_CAL_TOKENS=0 restores the whole corpus.
    CALTOK="${Q36_CAL_TOKENS:-1024}"
    case "$CALTOK" in 0|[1-9][0-9]*) ;; *) die "Q36_CAL_TOKENS must be 0 or a positive integer (got '$CALTOK')";; esac
    PROMPT="$CORPUS"
    if [ "$CALTOK" != "0" ]; then
      PROMPT="$(mktemp /tmp/q36cal.XXXXXX)"
      head -c $((CALTOK * 4)) "$CORPUS" > "$PROMPT" || die "head -c failed on $CORPUS"
      BYTES=$(wc -c < "$PROMPT")
      [ "$BYTES" -ge 4 ] || die "calib prompt ended up empty ($BYTES bytes) — is Q36_CORPUS='$CORPUS' readable and non-empty?"
      echo "[minipc] calibrate: corpus capped at ~$CALTOK tokens ($BYTES bytes taken, Q36_CAL_TOKENS=0 for the full file)"
    fi
    # ETA banner: measured rate from the last calibrate on THIS box if present;
    # otherwise max(compute, weight-streaming I/O) — at HOT_FRAC=1.0 every token
    # re-streams the whole trunk, which on an 8 GB box dominates compute.
    python3 - "$MODEL" "$THREADS" "$PROMPT" <<'PY'
import glob, json, os, sys, time
model, threads, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    toks = max(1, os.path.getsize(prompt) // 4)
    rate_f = os.path.join(model, ".q36_cal_rate")
    if os.path.exists(rate_f):
        spt, when = open(rate_f).read().split()
        spt, when = float(spt), float(when)
        age = (time.time() - when) / 86400
        print("[minipc] ETA: ~%d tokens x %.2f s/token (MEASURED on this box %.1f d ago) => ~%.0f min"
              % (toks, spt, age, toks * spt / 60))
    else:
        cfg = json.load(open(os.path.join(model, "config.json")))
        H, L, I = cfg.get("hidden_size", 0), cfg.get("num_hidden_layers", 0), cfg.get("intermediate_size", 0)
        P = L * (4 * H * H + 3 * H * I)
        spt_c = 2 * P / (threads * 6e9)
        trunk = sum(os.path.getsize(f) for f in glob.glob(os.path.join(model, "*.safetensors")))
        bw_env = os.environ.get("Q36_ASSUME_BW_MBPS")
        if bw_env:
            bw = float(bw_env) * 1e6
        else:
            rot = False
            import glob as _g
            for rf in _g.glob("/sys/block/*/queue/rotational"):
                try: rot = rot or open(rf).read().strip() == "1"
                except OSError: pass
            bw = (150 if rot else 1500) * 1e6
        one_stream = trunk / bw            # prefill streams the trunk ONCE for all N tokens;
        total = toks * spt_c + 17 * one_stream   # the 16 decode tokens re-stream it one-by-one
        print("[minipc] ETA: >= ~%.0f min  (first-run model: %d tokens x %.1f s compute = %.0f min, + %d trunk streams x %.0f s = %.1f min at %.0f MB/s)"
              % (total / 60, toks, spt_c, toks * spt_c / 60, 17, one_stream, 17 * one_stream / 60, bw / 1e6))
        print("[minipc]   Q36_ASSUME_BW_MBPS adjusts the I/O term; the real rate is cached after this run")
    print("[minipc] plus a one-time trunk-warm pass before the first progress line (cold cache: minutes),")
    print("[minipc] and a SILENT tail after the bar hits 100%: 16 decode tokens, each re-streaming the trunk")
    try:
        L = max(1, int(json.load(open(os.path.join(model, "qwen36_meta.json"))).get("n_layers", 0)
                       or json.load(open(os.path.join(model, "config.json"))).get("num_hidden_layers", 1)))
    except Exception:
        L = 64
    gap = toks * (2 * P / (threads * 6e9)) / L + (trunk / L) / bw
    print("[minipc] the bar updates ONCE PER LAYER: expect ~%.0f s between lines at %d tokens (%d layers)."
          % (gap, toks, L))
    print("[minipc] minutes of silence mid-bar are normal — check top/iostat before calling it stuck")
    print("[minipc] too long? Q36_CAL_TOKENS=<n> caps it (the energy ranking converges early); 0 = full corpus")
except Exception as e:
    print("[minipc] ETA: unknown (%s)" % e)
PY
    [ -f "$OUT" ] && echo "[minipc] note: existing profile $OUT is overwritten at END of run only"
    echo "[minipc] full-compute prefill now runs; engine stderr streams live below;"
    echo "[minipc] the hot profile is written ONLY at clean exit — bound the run, do not ^C mid-pass"
    env SNAP=$MODEL Q36_MAXT=$CTX OMP_NUM_THREADS=$THREADS COLI_NO_OMP_TUNE=1 \
        TRUNK_MAP=1 HOT_FRAC=1.0 KV8=$KV8V HOT_PROFILE=$OUT N_NEW=16 COLI_TEMP=0 \
        Q36_TRUNK_STREAM="${Q36_TRUNK_STREAM:-1}" \
        "$ENGINE" "$CAP" 4 "$PROMPT" > /dev/null 2> >(tee /tmp/calib.err | q36_render >&2)
    RC=$?
    wait 2>/dev/null || true   # let the tee drain
    grep -o "\[hot\].*" /tmp/calib.err || true
    [ "$RC" -eq 0 ] || die "calibrate run rc=$RC — last lines above, full log /tmp/calib.err"
    [ -f "$OUT" ] || die "no profile at $OUT — killed early? it lands only at clean exit (cap with Q36_CAL_TOKENS)"
    python3 - "$MODEL" <<'PY'
import os, re, sys, time
err = "/tmp/calib.err"
txt = open(err, errors="replace").read() if os.path.exists(err) else ""
m = re.search(r"\[timers\] prefill: (\d+) tokens\s+dn=([\d.]+) attn=([\d.]+) moe=([\d.]+)\(sh=[\d.]+ rt=[\d.]+\) head=([\d.]+) ms", txt)
if m:
    toks = int(m.group(1))
    ms = sum(float(m.group(i)) for i in (2, 3, 4, 5))
    if toks > 0:
        open(os.path.join(sys.argv[1], ".q36_cal_rate"), "w").write("%.4f %d" % (ms / 1000.0 / toks, time.time()))
        print("[minipc] measured %.2f s/token cached in .q36_cal_rate — next ETA is measured, not modeled" % (ms / 1000.0 / toks))
PY
    echo "[minipc] profile: $OUT  (now run: q36-minipc sort <model> <profile>)"
    ;;
  sort)
    # dense: permute the container by the profile -> hotness-sorted copy
    [ "$CLASS" = dense ] || die "sort is for dense containers"
    PROF="${ARG3:-$MODEL/hot_profile.json}"
    [ -f "$PROF" ] || die "profile not found: $PROF (run calibrate first)"
    OUT="${Q36_SORT_OUT:-$MODEL-sorted}"
    python3 "$TOOLS/hot_sort.py" "$MODEL" "$PROF" "$OUT"
    echo "[minipc] sorted container: $OUT"
    echo "[minipc] serving it: Q36_GRIFFIN=0 (use the static prefix) or keep GRIFFIN=1 (order no longer matters)"
    ;;
  watch)
    echo "[minipc] live bar from /tmp/calib.err (Ctrl-C quits; run calibrate/bench in another terminal)"
    tail -n +1 -f /tmp/calib.err 2>/dev/null | q36_render
    ;;
  doctor)
    echo "== engine =="; ls -la "$ENGINE" | awk '{print $5, $9}'
    echo "== knobs present in binary =="; for k in TRUNK_MAP GRIFFIN HOT_FRAC TRUNK_WARM HOT_PROFILE HOT_LAYERS KV8 KV_TQ KV_TQ_CODEC Q36_FLASH Q36_FAST_SOFTMAX; do
      c=$(grep -c "$k" "$HOME/colibri/c/qwen36.c"); echo "  $k: $c refs"
    done
    echo "== model: $MODEL ($CLASS, num_experts=$NEXP)"
    if [ -f "$MODEL/qwen36_meta.json" ]; then
      grep -q '"mtp"' "$MODEL/qwen36_meta.json" && echo "== mtp head: container carries meta.mtp (Phase 2b will use it)" || echo "== mtp head: none in container meta"
    else
      echo "== mtp head: no qwen36_meta.json found"
    fi
    echo "== level: $LEVEL (cap=$CAP threads=$THREADS ctx=$CTX kv8=$KV8V frac=$FRAC)"
    echo "== ram: $(free -h | awk '/Mem:/ {print $2}')"
    ;;
  help|*)
    cat <<EOF
q36-minipc v2 - one-knob qwen36 launcher for small boxes (MoE AND dense)
  q36-minipc serve     <model_dir> [port]   long-running serve (auto class)
  q36-minipc chat      <model_dir>          one-shot chat on stdin
  q36-minipc bench     <model_dir> [prompt] honest battery
  q36-minipc seed      <model_dir>          (MoE) Zipf .coli_usage prior
  q36-minipc calibrate <model_dir>          (dense) HOT_PROFILE run; Q36_CORPUS=<file>
  q36-minipc sort      <model_dir> [prof]   (dense) hot-sort a copy via hot_sort.py
  q36-minipc doctor    <model_dir>          knob/model/level report
  q36-minipc watch                          live progress bar for a running calibrate/bench
Env: Q36_LEVEL=eco|balanced|quality · Q36_KVQ=f32|kv8|tq4|tq3 (KV tier) · Q36_HOT_FRAC=<f> · Q36_GRIFFIN=0|1 (dense)
     Q36_HOT_LAYERS=<file> · Q36_CORPUS=<file> · Q36_SORT_OUT=<dir> · Q36_RAM_GB=<n>
KV tiers: f32 exact · kv8 fp8 (default) · tq4 rot-int4 ~0.10 relL2 · tq3 PolarQuant-3 (RAM extreme)
Dense profile: TRUNK_MAP + GRIFFIN + HOT_FRAC + TRUNK_WARM + KV tier (frac 1.0 = exact full model)
MoE profile:   cap + PIN=auto + PILOT=2 + WIDE + KV tier (exact routing, full capability)
EOF
    ;;
esac
