#!/usr/bin/env bash
# run_battery.sh — the 2b-4 measurement battery, one command, structured output.
#
#   tools/run_battery.sh <model_dir> <prompt_file> [out_json]
#
# Arms (all greedy, COLI_TEMP=0, honest [timers] decode numbers):
#   A f32            B kv8             C tq4 absorb=0      D tq4 absorb=1
#   E kv8+MTP drafts2  F kv8+MTP forced-accept (ref.json bootstrapped in-run)
# Per arm: decode ms/token ([timers] TOTAL), TTFT s, PEAK RSS GB, rc, and the
# [qwen36 MTP] report line where present. Writes results/battery_<host>_<ts>.json
# (or $3) and prints a summary table. Env: BATTERY_THREADS (default 4),
# BATTERY_NNEW (default 64), BATTERY_CAP (default 8), BATTERY_ARMS="A B C..."
# subset, BATTERY_TIMEOUT (per-arm wall seconds, default 3600, 0 = unlimited).
#
# No engine changes; runs the built binary directly. Cost = one prompt run per
# requested arm (9 in the default set; BATTERY_ARMS="A B" runs only those).
#
# E11: the runner invokes the BINARY, so every env name here must be one the
# engine itself getenv()s (scripts/check_runner_env.py enforces this in CI).
# The launcher's Q36_* aliases (Q36_HOT_FRAC, Q36_GRIFFIN) are mapped by
# q36-minipc.sh and are INVISIBLE to the binary — setting the alias alone runs
# the stock default (frac 1.0, griffin off). Dense containers also need
# TRUNK_MAP=1 (weights stay mmap'd on disk; without it the loader pulls the
# whole ~50 GB F16 trunk into RAM and an 8 GB box OOM-kills it at its ~5 GB
# watermark — that was the rc=137 signature in battery_llm-030_20261023).
# Env: BATTERY_FRAC (engine HOT_FRAC — exactness/pruning knob: 1.0 = exact,
# 0.25 = mini-PC daily point; on an UNSORTED container <1.0 prunes arbitrary
# neurons: mechanics valid, quality meaningless until hot_sort), BATTERY_WARM
# (TRUNK_WARM, default 0 = cold honest numbers; the W read-matrix wants cold),
# BATTERY_STREAM (Q36_TRUNK_STREAM=1 on rotating disks — E9 thrash control,
# bit-identical ids per its CI gate).
#
# E12 (2026-10-24): the first post-E11 field runs PROVED the OOM fix (no
# rc=137; peak RSS ~7.0 GB of 7.62; trunk-stream effective 10-12 MB/s) and
# then every compute arm died rc=124 at this runner's hardcoded
# `timeout 3600`: fs_read 37-42 GB/arm ≈ exactly one cold prefill pass
# (0.75 × 54.7 GB at frac 0.25), so no arm ever finished prefill — the old
# "~3-5 h overnight" guidance was ~10x off (prefill floor alone is ~60-90
# min/arm on a rotating disk, and frac-0.25 decode re-reads GBs per token).
# BATTERY_TIMEOUT is now a knob (0 = unlimited), per-arm logs persist beside
# the JSON (${OUT%.json}_logs/) instead of dying in /tmp, and the runner
# prints an I/O-budget preflight. HDD plan: probe ONE arm to completion
# first — BATTERY_ARMS="A" BATTERY_NNEW=2 BATTERY_TIMEOUT=0 — read its real
# TTFT + ms/token from the JSON, then budget the matrix (ROADMAP E12).
set -uo pipefail
MODEL="${1:?usage: run_battery.sh <model_dir> <prompt_file> [out_json]}"
PF="${2:?usage: run_battery.sh <model_dir> <prompt_file> [out_json]}"
TS=$(date +%Y%m%d-%H%M%S)
OUT="${3:-results/battery_$(hostname -s)_$TS.json}"
THR="${BATTERY_THREADS:-4}"; NNEW="${BATTERY_NNEW:-64}"; CAP="${BATTERY_CAP:-8}"
ARMS="${BATTERY_ARMS:-A B C D E F W0 W256 W1G}"
FRAC="${BATTERY_FRAC:-1.0}"; MAXT="${BATTERY_MAXT:-16384}"
WARM="${BATTERY_WARM:-0}"; STREAM="${BATTERY_STREAM:-0}"
TMO="${BATTERY_TIMEOUT:-3600}"
case "$TMO" in ''|*[!0-9]*) echo "run_battery: BATTERY_TIMEOUT must be a non-negative integer (seconds; 0 = unlimited), got '$TMO'" >&2; exit 2 ;; esac
if [ "$TMO" != 0 ]; then TMO_CMD="timeout $TMO"; else TMO_CMD=""; fi
RAMGB="${BATTERY_RAM_GB:-$(awk '/MemTotal/ {printf "%d", $2/1048576+0.5}' /proc/meminfo 2>/dev/null || echo 8)}"
PB=$(wc -c < "$PF")
if [ "$PB" -gt "$MAXT" ]; then
  echo "run_battery: prompt is ~$PB tokens (byte-level tokenizer) but Q36_MAXT=$MAXT." >&2
  echo "  raise BATTERY_MAXT, or use a shorter prompt. 8 GB mini-PC profile:" >&2
  echo "  BATTERY_FRAC=0.25 BATTERY_MAXT=8192 with a <=8k-token prompt" >&2
  echo "  (e.g. head -c 7900 ~/eval/prompt_8k.txt > ~/eval/prompt_7k.txt)" >&2
  exit 2
fi
W=/tmp/battery_$TS; mkdir -p "$W"
# engine-native names only (see E11 header); TRUNK_MAP=1 is mandatory for dense
# containers on small-RAM boxes — stock-dialect runs OOM (rc=137) mid-load.
COMMON="SNAP=$MODEL Q36_MAXT=$MAXT N_NEW=$NNEW COLI_TEMP=0 COLI_TIMERS=1 COLI_NO_OMP_TUNE=1 KVSAVE=0 OMP_NUM_THREADS=$THR TRUNK_MAP=1 TRUNK_WARM=$WARM HOT_FRAC=$FRAC RAM_GB=$RAMGB"
[ "$STREAM" = 1 ] && COMMON="$COMMON Q36_TRUNK_STREAM=1"
ENGINE="${Q36_ENGINE:-$HOME/colibri/c/qwen36}"
echo "[battery] engine=$ENGINE frac=$FRAC maxt=$MAXT warm=$WARM stream=$STREAM ram_gb=$RAMGB threads=$THR n_new=$NNEW"
echo "[battery] TRUNK_MAP=1: trunk stays mmap'd on disk (page cache + kernel eviction, no OOM)."
# E12 I/O-budget preflight: cold bytes per full pass vs the measured field rate.
MODEL_MB=$(du -sm "$MODEL" 2>/dev/null | cut -f1); MODEL_MB=${MODEL_MB:-0}
if [ "$MODEL_MB" -gt 0 ]; then
  COLD_MB=$(awk -v m="$MODEL_MB" -v f="$FRAC" 'BEGIN{printf "%d", m*(1-f)}')
  echo "[battery] E12 budget: model ${MODEL_MB} MB, frac ${FRAC} -> ~${COLD_MB} MB cold reads per full pass;"
  echo "[battery]   at the llm-030 measured 10-12 MB/s that is a ~$((COLD_MB/12/60+1))-$((COLD_MB/10/60+1)) min prefill floor PER ARM before token 1."
  echo "[battery]   BATTERY_TIMEOUT=${TMO}s per arm (0 = unlimited). rc=124 in the JSON = killed at this wall, not an engine fault."
fi

run_arm() {  # run_arm <name> <extra env...> -- prompt
  local name=$1; shift
  echo "== arm $name: $*"
  /usr/bin/time -v env $COMMON "$@" $TMO_CMD "$ENGINE" "$CAP" "$THR" "$PF" > "$W/$name.out" 2> "$W/$name.err"
  echo $? > "$W/$name.rc"
}
want() { case " $ARMS " in *" $1 "*) return 0 ;; *) echo "== arm $1: not in BATTERY_ARMS, skipped" ; return 1 ;; esac; }
grep -q '"mtp"' "$MODEL/qwen36_meta.json" 2>/dev/null && HAS_MTP=1 || HAS_MTP=0
[ "$HAS_MTP" = 1 ] || echo "== arms E/F will skip: container carries no MTP head (scripts/gen_mtp_head_synth.py adds one)"
want A && run_arm A
want B && run_arm B KV8=1
want C && run_arm C KV_TQ=4 KV_TQ_CODEC=1 Q36_TQ_ABSORB=0
want D && run_arm D KV_TQ=4 KV_TQ_CODEC=1 Q36_TQ_ABSORB=1
if [ "$HAS_MTP" = 1 ]; then
want E && run_arm E KV8=1 Q36_MTP=1 Q36_MTP_DRAFTS=2 COLI_SPEC_GATE=0 COLI_LOOKUP=0
# F: bootstrap the oracle from arm A's id stream, then force-accept
if want F; then
if [ -f "$W/A.ids" ] || DUMP_IDS="$W/A.ids" env $COMMON $TMO_CMD "$ENGINE" "$CAP" "$THR" "$PF" > /dev/null 2> "$W/A_ids.err"; then
  python3 - "$W/A.ids" "$W/ref.json" "$NNEW" <<'PY'
import json, sys
ids = [int(l) for l in open(sys.argv[1])]
n = int(sys.argv[3])
json.dump({"prompt_ids": ids[:-n], "full_ids": ids}, open(sys.argv[2], "w"))
print("ref.json: prompt %d full %d" % (len(ids) - n, len(ids)))
PY
  run_arm F KV8=1 Q36_MTP=1 Q36_MTP_FORCE=accept COLI_SPEC_GATE=0 COLI_LOOKUP=0 Q36_MTP_DRAFTS=2
else
  echo "!! could not bootstrap ref.json; skipping arm F"; echo 9 > "$W/F.rc"
fi
fi
else
  echo "== arm E: skipped (no MTP head)"; echo 0 > "$W/E.rc"; echo "skipped: no mtp head" > "$W/E.err"
  echo "== arm F: skipped (no MTP head)"; echo 0 > "$W/F.rc"; echo "skipped: no mtp head" > "$W/F.err"
fi

# window x GRIFFIN matrix (Phase 4b slice 3): read-bytes via /usr/bin/time -v.
# Fixed pruned point (frac 0.25) — the matrix varies WIN_MB only; cold cache
# (BATTERY_WARM=0 default) is what makes fs_read_gb comparable across W arms.
want W0 && run_arm W0   HOT_FRAC=0.25 GRIFFIN=1 Q36_WIN_MB=0
want W256 && run_arm W256 HOT_FRAC=0.25 GRIFFIN=1 Q36_WIN_MB=256
want W1G && run_arm W1G  HOT_FRAC=0.25 GRIFFIN=1 Q36_WIN_MB=1024

# E12: persist the per-arm forensics beside the JSON — /tmp dies on reboot and
# the 2026-10-24 runs' progress lines (how far each arm got) were lost that way.
LOGD="${OUT%.json}_logs"; mkdir -p "$LOGD"
for f in "$W"/*.err "$W"/*.out "$W"/*.rc; do [ -e "$f" ] && cp "$f" "$LOGD/"; done

python3 - "$W" "$OUT" "$MODEL" "$PF" "$THR" "$NNEW" "$ARMS" "$FRAC" "$MAXT" "$WARM" "$STREAM" "$RAMGB" "$ENGINE" "$TMO" <<'PY'
import json, os, re, sys, time
W, OUT, MODEL, PF, THR, NNEW, ARMS = sys.argv[1:8]
FRAC, MAXT, WARM, STREAM, RAMGB, ENGINE, TMO = (float(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]),
                                                int(sys.argv[11]), int(sys.argv[12]), sys.argv[13],
                                                int(sys.argv[14]))
def rd(n):
    p = os.path.join(W, n)
    return open(p, errors="replace").read() if os.path.exists(p) else ""
arms = {}
for a in ARMS.split():
    err = rd(a + ".err")
    rc = rd(a + ".rc").strip()
    tot = re.search(r"TOTAL\s+[\d.]+ ms\s+([\d.]+) ms/token", err)
    ttft = re.search(r"TTFT[^\d]*([\d.]+)\s*s", err)
    if not ttft:
        pf = re.search(r"prefill: \d+ tokens\s+dn=([\d.]+) attn=([\d.]+) moe=([\d.]+).*?head=([\d.]+) ms", err)
        if pf:
            arms_tmp = sum(float(g) for g in pf.groups()) / 1000.0
        else:
            arms_tmp = None
    else:
        arms_tmp = float(ttft.group(1))
    rss = re.search(r"PEAK RSS:\s*([\d.]+) GB", err)
    fsb = re.search(r"File system inputs:\s*(\d+)", err)
    # /usr/bin/time -v reports even when the child was killed — the OOM forensics
    trss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", err)
    term = re.search(r"Command terminated by signal (\d+)", err)
    mtp = [l for l in err.splitlines() if "[qwen36 MTP] run:" in l]
    arms[a] = {"rc": int(rc or -1),
               "decode_ms_per_tok": float(tot.group(1)) if tot else None,
               "ttft_s": arms_tmp,
               "peak_rss_gb": float(rss.group(1)) if rss else (round(int(trss.group(1)) / 1048576.0, 3) if trss else None),
               "rss_source": "engine" if rss else ("time_v" if trss else None),
               "fs_read_gb": round(int(fsb.group(1)) * 512 / 1e9, 3) if fsb else None,
               "terminated_by_signal": int(term.group(1)) if term else None,
               "mtp_report": mtp[-1] if mtp else None}
def meminfo_gb(key):
    try:
        for l in open("/proc/meminfo"):
            if l.startswith(key):
                return round(int(l.split()[1]) / 1048576.0, 2)
    except OSError:
        pass
    return None
model_bytes = sum(os.path.getsize(os.path.join(MODEL, f))
                  for f in os.listdir(MODEL) if f.endswith(".safetensors")) \
    if os.path.isdir(MODEL) else 0
doc = {"host": os.uname().nodename, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "model": MODEL, "prompt": PF, "threads": int(THR), "n_new": int(NNEW),
       "operating_point": {"engine": ENGINE, "hot_frac": FRAC, "maxt": MAXT,
                           "trunk_map": 1, "trunk_warm": WARM, "trunk_stream": STREAM,
                           "timeout_s": TMO or None,
                           "ram_gb": RAMGB, "model_gb": round(model_bytes / 1e9, 1),
                           "mem_total_gb": meminfo_gb("MemTotal"),
                           "mem_available_gb_at_end": meminfo_gb("MemAvailable")},
       "note": "greedy COLI_TEMP=0; decode = [timers] TOTAL ms/token (Speed: line includes prefill, never quoted); rc=137 + terminated_by_signal=9 = OOM kill (peak_rss_gb then from /usr/bin/time -v); rc=124 = killed at the BATTERY_TIMEOUT wall mid-run — null metrics mean 'never finished', not 'engine fault' (E12)",
       "arms": arms}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(doc, open(OUT, "w"), indent=1)
print("\n== battery summary (%s)" % OUT)
print("arm   rc  decode ms/tok  ttft s  RSS GB  fsRead GB  mtp/win")
for a, d in arms.items():
    print("%-5s %-3s %-14s %-7s %-7s %-10s %s" % (
        a, d["rc"], d["decode_ms_per_tok"], d["ttft_s"], d["peak_rss_gb"], d["fs_read_gb"],
        (d["mtp_report"] or "")[:48]))
PY
echo "[battery] wrote $OUT + per-arm logs in $LOGD — commit BOTH, paste the summary"
