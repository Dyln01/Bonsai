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
# BATTERY_NNEW (default 64), BATTERY_CAP (default 8), BATTERY_ARMS="A B C..." subset.
#
# No engine changes; runs the built binary directly. ~6 runs x your prompt cost.
set -uo pipefail
MODEL="${1:?usage: run_battery.sh <model_dir> <prompt_file> [out_json]}"
PF="${2:?usage: run_battery.sh <model_dir> <prompt_file> [out_json]}"
TS=$(date +%Y%m%d-%H%M%S)
OUT="${3:-results/battery_$(hostname -s)_$TS.json}"
THR="${BATTERY_THREADS:-4}"; NNEW="${BATTERY_NNEW:-64}"; CAP="${BATTERY_CAP:-8}"
ARMS="${BATTERY_ARMS:-A B C D E F W0 W256 W1G}"
FRAC="${BATTERY_FRAC:-1.0}"; MAXT="${BATTERY_MAXT:-16384}"
PB=$(wc -c < "$PF")
if [ "$PB" -gt "$MAXT" ]; then
  echo "run_battery: prompt is ~$PB tokens (byte-level tokenizer) but Q36_MAXT=$MAXT." >&2
  echo "  raise BATTERY_MAXT, or use a shorter prompt. 8 GB mini-PC profile:" >&2
  echo "  BATTERY_FRAC=0.25 BATTERY_MAXT=8192 with a <=8k-token prompt" >&2
  echo "  (e.g. head -c 7900 ~/eval/prompt_8k.txt > ~/eval/prompt_7k.txt)" >&2
  exit 2
fi
W=/tmp/battery_$TS; mkdir -p "$W"
COMMON="SNAP=$MODEL Q36_MAXT=$MAXT N_NEW=$NNEW COLI_TEMP=0 COLI_TIMERS=1 COLI_NO_OMP_TUNE=1 KVSAVE=0 OMP_NUM_THREADS=$THR Q36_HOT_FRAC=$FRAC"
ENGINE="${Q36_ENGINE:-$HOME/colibri/c/qwen36}"

run_arm() {  # run_arm <name> <extra env...> -- prompt
  local name=$1; shift
  echo "== arm $name: $*"
  /usr/bin/time -v env $COMMON "$@" timeout 3600 "$ENGINE" "$CAP" "$THR" "$PF" > "$W/$name.out" 2> "$W/$name.err"
  echo $? > "$W/$name.rc"
}
grep -q '"mtp"' "$MODEL/qwen36_meta.json" 2>/dev/null && HAS_MTP=1 || HAS_MTP=0
[ "$HAS_MTP" = 1 ] || echo "== arms E/F will skip: container carries no MTP head (scripts/gen_mtp_head_synth.py adds one)"
run_arm A
run_arm B KV8=1
run_arm C KV_TQ=4 KV_TQ_CODEC=1 Q36_TQ_ABSORB=0
run_arm D KV_TQ=4 KV_TQ_CODEC=1 Q36_TQ_ABSORB=1
if [ "$HAS_MTP" = 1 ]; then
run_arm E KV8=1 Q36_MTP=1 Q36_MTP_DRAFTS=2 COLI_SPEC_GATE=0 COLI_LOOKUP=0
# F: bootstrap the oracle from arm A's id stream, then force-accept
if [ -f "$W/A.ids" ] || DUMP_IDS="$W/A.ids" env $COMMON timeout 3600 "$ENGINE" "$CAP" "$THR" "$PF" > /dev/null 2> "$W/A_ids.err"; then
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
else
  echo "== arm E: skipped (no MTP head)"; echo 0 > "$W/E.rc"; echo "skipped: no mtp head" > "$W/E.err"
  echo "== arm F: skipped (no MTP head)"; echo 0 > "$W/F.rc"; echo "skipped: no mtp head" > "$W/F.err"
fi

# window x GRIFFIN matrix (Phase 4b slice 3): read-bytes via /usr/bin/time -v
run_arm W0   HOT_FRAC=0.25 GRIFFIN=1 Q36_WIN_MB=0
run_arm W256 HOT_FRAC=0.25 GRIFFIN=1 Q36_WIN_MB=256
run_arm W1G  HOT_FRAC=0.25 GRIFFIN=1 Q36_WIN_MB=1024

python3 - "$W" "$OUT" "$MODEL" "$PF" "$THR" "$NNEW" "$ARMS" <<'PY'
import json, os, re, sys, time
W, OUT, MODEL, PF, THR, NNEW, ARMS = sys.argv[1:8]
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
    mtp = [l for l in err.splitlines() if "[qwen36 MTP] run:" in l]
    arms[a] = {"rc": int(rc or -1),
               "decode_ms_per_tok": float(tot.group(1)) if tot else None,
               "ttft_s": arms_tmp,
               "peak_rss_gb": float(rss.group(1)) if rss else None,
               "fs_read_gb": round(int(fsb.group(1)) * 512 / 1e9, 3) if fsb else None,
               "mtp_report": mtp[-1] if mtp else None}
doc = {"host": os.uname().nodename, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "model": MODEL, "prompt": PF, "threads": int(THR), "n_new": int(NNEW),
       "note": "greedy COLI_TEMP=0; decode = [timers] TOTAL ms/token (Speed: line includes prefill, never quoted)",
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
echo "[battery] raw stderr/stdout per arm in $W — commit $OUT + paste the summary"
