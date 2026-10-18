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
        "$ENGINE" "$CAP" 4 "$PF" 2>&1 | \
      grep -E "== qwen36|RSS after load|hit rate|TTFT|decode: [0-9]+ tokens|TOTAL|PEAK RSS|Speed:|PIN|USAGE|warm|hot]"
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
    env SNAP=$MODEL Q36_MAXT=$CTX OMP_NUM_THREADS=$THREADS COLI_NO_OMP_TUNE=1 \
        TRUNK_MAP=1 HOT_FRAC=1.0 KV8=$KV8V HOT_PROFILE=$OUT N_NEW=16 COLI_TEMP=0 \
        "$ENGINE" "$CAP" 4 "$CORPUS" > /dev/null 2> /tmp/calib.err
    grep -o "\[hot\].*" /tmp/calib.err || true
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
Env: Q36_LEVEL=eco|balanced|quality · Q36_KVQ=f32|kv8|tq4|tq3 (KV tier) · Q36_HOT_FRAC=<f> · Q36_GRIFFIN=0|1 (dense)
     Q36_HOT_LAYERS=<file> · Q36_CORPUS=<file> · Q36_SORT_OUT=<dir> · Q36_RAM_GB=<n>
KV tiers: f32 exact · kv8 fp8 (default) · tq4 rot-int4 ~0.10 relL2 · tq3 PolarQuant-3 (RAM extreme)
Dense profile: TRUNK_MAP + GRIFFIN + HOT_FRAC + TRUNK_WARM + KV tier (frac 1.0 = exact full model)
MoE profile:   cap + PIN=auto + PILOT=2 + WIDE + KV tier (exact routing, full capability)
EOF
    ;;
esac
