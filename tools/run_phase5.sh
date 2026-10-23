#!/usr/bin/env bash
# run_phase5.sh — the real-weights validation session (docs/VALIDATION_PROTOCOL_QWEN36.md).
#
#   tools/run_phase5.sh <container_dir> <wiki.test.raw> [session_dir]
#
# Resumable: every arm writes <session>/arms/<arm>.err; an arm whose .err exists
# is skipped unless PHASE5_FORCE=1. Interrupt between arms freely.
#
# Arms (ppl via the engine's PPL=1 tf-nll on the wikitest ref.json), bands
# MIRRORED from docs/VALIDATION_PROTOCOL_QWEN36.md §3 — update both together:
#   A frac1.0 griffin0        B frac0.5        C frac0.25       D C+griffin1
#   E C+sorted (needs M3 calibrate+sort first) F C+KV8          G C+TQ4-absorb
#   H C+FAST_SOFTMAX          I W2+MTP (skipped unless meta.mtp)
# Mechanics: M1 determinism (arm A twice, DUMP_IDS compare), M3 calibrate+sort
# (PHASE5_M3=1), M4 RSS from every arm's PEAK RSS line, M5 battery arms E/F
# (auto when meta.mtp), M2 prefix-reuse skipped here (needs your agent-20
# transcripts: run scripts/exp_prefix_q36.py separately and drop its JSON in).
#
# Env: PHASE5_LEN (default 2048 scored tokens — mini-PC budget; bands unchanged),
#      PHASE5_CTX (512), PHASE5_THREADS (4), PHASE5_CAP (8), PHASE5_FORCE, PHASE5_M3.
#
# E11: arms must use ENGINE-NATIVE env names (HOT_FRAC, GRIFFIN — not the
# launcher's Q36_* aliases, which the binary never reads; scripts/
# check_runner_env.py enforces this in CI). Pre-E11 sessions that ran arms
# with the alias names measured frac 1.0/griffin 0 on EVERY arm: rerun them
# with PHASE5_FORCE=1 (the resume cache will otherwise skip the tainted .err).
set -uo pipefail
CONT="${1:?usage: run_phase5.sh <container_dir> <wiki.test.raw> [session_dir]}"
WIKI="${2:?usage: run_phase5.sh <container_dir> <wiki.test.raw> [session_dir]}"
TS=$(date +%Y%m%d-%H%M)
SES="${3:-results/realweights_$TS}"
mkdir -p "$SES/arms"
THR="${PHASE5_THREADS:-4}"; CAP="${PHASE5_CAP:-8}"; LEN="${PHASE5_LEN:-2048}"; CTX="${PHASE5_CTX:-512}"
ENGINE="${Q36_ENGINE:-$HOME/colibri/c/qwen36}"
COMMON="SNAP=$CONT PPL=1 COLI_TEMP=0 COLI_NO_OMP_TUNE=1 KVSAVE=0 OMP_NUM_THREADS=$THR TRUNK_MAP=1"

echo "== phase 5 session: $SES"
echo "== preflight"
[ -f "$CONT/config.json" ] || { echo "no config.json in $CONT — convert the checkpoint with colibri's converter first"; exit 2; }
[ -f "$CONT/tokenizer.json" ] || { echo "no tokenizer.json in $CONT"; exit 2; }
bash "$(dirname "$0")/verify_engine.sh" | tail -1 || exit 2
find "$CONT" -name '*.safetensors' -exec sha256sum {} \; | tee "$SES/manifest.txt"
echo "engine: $(md5sum "$ENGINE" | cut -d' ' -f1)  threads=$THR len=$LEN ctx=$CTX"

if [ ! -f "$SES/ref.json" ]; then
  python3 "$(dirname "$0")/../scripts/make_wikitest_ref.py" "$CONT" "$WIKI" "$SES/ref.json" --ctx=$CTX --len=$LEN || exit 2
fi

arm() {  # arm <name> <extra env...>
  local name=$1; shift
  if [ -f "$SES/arms/$name.err" ] && [ "${PHASE5_FORCE:-0}" != 1 ]; then echo "== arm $name: cached"; return; fi
  echo "== arm $name: $*  (started $(date +%H:%M:%S))"
  env $COMMON "$@" "$ENGINE" "$CAP" "$THR" "$SES/ref.json" > "$SES/arms/$name.out" 2> "$SES/arms/$name.err"
  echo "rc=$?" >> "$SES/arms/$name.err"
  echo "== arm $name: done $(date +%H:%M:%S)"
}
arm A HOT_FRAC=1.0 GRIFFIN=0
arm B HOT_FRAC=0.5 GRIFFIN=0
arm C HOT_FRAC=0.25 GRIFFIN=0
arm D HOT_FRAC=0.25 GRIFFIN=1
if [ "${PHASE5_M3:-0}" = 1 ] && [ ! -d "$CONT-sorted" ]; then
  echo "== M3: calibrate (bounded) + sort"
  Q36_CORPUS="${PHASE5_CORPUS:-$CONT/calib_prompt.txt}" q36-minipc calibrate "$CONT" || echo "!! calibrate failed; arm E will skip"
  q36-minipc sort "$CONT" || echo "!! sort failed; arm E will skip"
fi
if [ -d "$CONT-sorted" ]; then arm E SNAP="$CONT-sorted" HOT_FRAC=0.25 GRIFFIN=0; else echo "== arm E: skipped (no $CONT-sorted; rerun with PHASE5_M3=1)"; fi
arm F HOT_FRAC=0.25 GRIFFIN=0 KV8=1
arm G HOT_FRAC=0.25 GRIFFIN=0 KV_TQ=4 KV_TQ_CODEC=1 Q36_TQ_ABSORB=1
arm H HOT_FRAC=0.25 GRIFFIN=0 Q36_FAST_SOFTMAX=1
if grep -q '"mtp"' "$CONT/qwen36_meta.json" 2>/dev/null; then
  arm I HOT_FRAC=0.25 GRIFFIN=0 Q36_MTP=1 Q36_MTP_DRAFTS=2
else
  echo "== arm I: skipped (container carries no MTP head; W2 + gen_mtp_head_synth.py to enable)"
fi
echo "== M1: determinism (arm A rerun with DUMP_IDS vs first-pass ids)"
DUMP_IDS="$SES/arms/A.ids" env $COMMON HOT_FRAC=1.0 GRIFFIN=0 "$ENGINE" "$CAP" "$THR" "$SES/ref.json" > /dev/null 2> "$SES/arms/A_det.err" || true
DUMP_IDS="$SES/arms/A2.ids" env $COMMON HOT_FRAC=1.0 GRIFFIN=0 "$ENGINE" "$CAP" "$THR" "$SES/ref.json" > /dev/null 2> "$SES/arms/A_det2.err" || true
cmp -s "$SES/arms/A.ids" "$SES/arms/A2.ids" && echo "M1 PASS: identical id streams" || echo "M1 FAIL: id streams differ"

python3 - "$SES" "$CONT" <<'PY'
import json, os, re, sys, time
ses, cont = sys.argv[1], sys.argv[2]
BANDS = {"B": 0.02, "C": 0.05, "D": None, "E": None, "F": 0.01, "G": 0.02, "H": 0.002, "I": 0.005}
arms = {}
for f in sorted(os.listdir(os.path.join(ses, "arms"))):
    if not f.endswith(".err"): continue
    name = f[:-4]
    if name.startswith("A_") or name in ("A2",): continue
    base = os.path.join(ses, "arms", f[:-4])
    txt = ""
    for suf in (".out", ".err"):          # ppl/TF-NLL goes to stdout, timers/RSS to stderr
        if os.path.exists(base + suf):
            txt += open(base + suf, errors="replace").read()
    ppl = re.search(r"ppl = ([\d.]+)", txt)
    rss = re.search(r"PEAK RSS:\s*([\d.]+) GB", txt)
    rc = re.search(r"rc=(\d+)\s*$", txt)
    arms[name] = {"ppl": float(ppl.group(1)) if ppl else None,
                  "peak_rss_gb": float(rss.group(1)) if rss else None,
                  "rc": int(rc.group(1)) if rc else None}
base = (arms.get("A") or {}).get("ppl")
for name, d in arms.items():
    d["delta_rel_vs_A"] = round((d["ppl"] - base) / base, 4) if (d["ppl"] and base) else None
    band = BANDS.get(name)
    d["verdict"] = ("PASS" if band and d["delta_rel_vs_A"] is not None and abs(d["delta_rel_vs_A"]) <= band
                    else ("n/a" if name in ("A",) or band is None else "FAIL")) if d["ppl"] else "no-data"
doc = {"session": ses, "container": cont, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "scored_tokens": None, "arms": arms,
       "note": "bands mirrored from docs/VALIDATION_PROTOCOL_QWEN36.md §3; D/E judged vs C in the report"}
json.dump(doc, open(os.path.join(ses, "summary.json"), "w"), indent=1)
print("\n== session summary (%s/summary.json)" % ses)
print("arm  ppl       dRel(A)   RSS GB  rc  verdict")
for name, d in arms.items():
    print("%-4s %-9s %-9s %-7s %-3s %s" % (name, d["ppl"], d["delta_rel_vs_A"], d["peak_rss_gb"], d["rc"], d["verdict"]))
print("\nnext: commit %s (or tar it) and paste the summary — the report doc gets written from these numbers" % ses)
PY
echo "== session complete: $SES"
