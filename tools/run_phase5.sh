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
# Env: PHASE5_LEN (default 64 SCORED tokens — E16 field budget below; bands
#      unchanged), PHASE5_CHUNK (64 = scored tokens per weight pass; 1 = the
#      old per-token path), PHASE5_CTX (512), PHASE5_THREADS (4), PHASE5_CAP (8),
#      PHASE5_FORCE, PHASE5_M3, PHASE5_M1NEW (1 = M1 greedy steps),
#      PHASE5_REF_FRAC (0.25 = HOT_FRAC for the ref-build prefill only — the
#      ref ids are tokenizer output, forward-pass independent).
#
# Field budget (llm-030, E13-measured constants: trunk pass ~1600 s at frac
# 0.25 / ~5000 s at frac 1.0; lm_head ~90 s per scored token; prefill(512 ctx)
# ~0.6 h at frac 0.25 / ~1.4 h at frac 1.0):
#   per arm ≈ prefill(CTX) + ceil(LEN/CHUNK) × trunk pass + LEN × 90 s
#   defaults (LEN=64 CHUNK=64): ~2.7 h/arm at frac 0.25, ~4.4 h at frac 1.0;
#   full session (ref build + A B C D F G H + M1×2) ≈ 28–32 h continuous.
# E16: per-token tf-nll (CHUNK=1) is 2225 s/scored-token in the field — LEN
# 2048 would be ~52 DAYS per arm. Chunked scoring (Q36_TF_CHUNK, the spec-
# verify multi-row kernel, CI-parity-gated) is what makes Phase 5 runnable.
#
# E11: arms must use ENGINE-NATIVE env names (HOT_FRAC, GRIFFIN — not the
# launcher's Q36_* aliases, which the binary never reads; scripts/
# check_runner_env.py enforces this in CI). Pre-E11 sessions that ran arms
# with the alias names measured frac 1.0/griffin 0 on EVERY arm: rerun them
# with PHASE5_FORCE=1 (the resume cache will otherwise skip the tainted .err).
# E15: the ref build ALSO runs the engine — it must carry the same dialect
# (TRUNK_MAP=1 + a fitting HOT_FRAC), or the stock loader mallocs the whole
# 54.7 GB F16 trunk and an 8 GB box kills it in seconds. Three field sessions
# (realweights_20261026-0919/0949/1014) are manifest-only tombstones of that.
# E15b: M1 must run WITHOUT PPL=1 — the tf_nll branch returns before
# generate(), so DUMP_IDS would never be written and M1 could only report FAIL.
# E17: the first completed field arms (realweights_20261026-1014 A/B) ran on an
# engine whose map_f16_one lacked the F16->F32 exponent-bias rebase: EVERY
# mapped F16 weight read back 2^-112 of its true value, the trunk left the
# computation entirely, and tf scored ppl == vocab_size EXACTLY (logits ~1e-33,
# softmax uniform). Those arms/*.err are tombstones of that bug, not
# measurements. Rebuild the engine (tools/patch_mapf16_bias.py is in the
# chain), delete the session's arms/ (keep manifest.txt + ref.json — both
# engine-independent), rerun. The tripwire below aborts any session started on
# a pre-E17 engine right after its first arm instead of burning ~30 h of disk
# on dead numbers.
set -uo pipefail
CONT="${1:?usage: run_phase5.sh <container_dir> <wiki.test.raw> [session_dir]}"
WIKI="${2:?usage: run_phase5.sh <container_dir> <wiki.test.raw> [session_dir]}"
TS=$(date +%Y%m%d-%H%M)
SES="${3:-results/realweights_$TS}"
mkdir -p "$SES/arms"
THR="${PHASE5_THREADS:-4}"; CAP="${PHASE5_CAP:-8}"; LEN="${PHASE5_LEN:-64}"; CTX="${PHASE5_CTX:-512}"
CHUNK="${PHASE5_CHUNK:-64}"
ENGINE="${Q36_ENGINE:-$HOME/colibri/c/qwen36}"
COMMON="SNAP=$CONT PPL=1 COLI_TEMP=0 COLI_NO_OMP_TUNE=1 KVSAVE=0 OMP_NUM_THREADS=$THR TRUNK_MAP=1 Q36_TF_CHUNK=$CHUNK"
COMMON_GEN="SNAP=$CONT COLI_TEMP=0 COLI_NO_OMP_TUNE=1 KVSAVE=0 OMP_NUM_THREADS=$THR TRUNK_MAP=1"

echo "== phase 5 session: $SES"
echo "== preflight"
# E15: fail BEFORE the ~20-min manifest hash, not after — a missing wiki file
# killed nothing visibly three times; now it says so immediately.
[ -f "$WIKI" ] || { echo "no wiki.test.raw at '$WIKI' — fetch it first: python3 scripts/fetch_wikitest2_raw.py '$WIKI'"; exit 2; }
[ -f "$CONT/config.json" ] || { echo "no config.json in $CONT — convert the checkpoint with colibri's converter first"; exit 2; }
[ -f "$CONT/tokenizer.json" ] || { echo "no tokenizer.json in $CONT"; exit 2; }
bash "$(dirname "$0")/verify_engine.sh" | tail -1 || exit 2
if [ -f "$SES/manifest.txt" ]; then
  echo "manifest: cached ($(wc -l < "$SES/manifest.txt") shards) — E15 resume"
else
  find "$CONT" -name '*.safetensors' -exec sha256sum {} \; | tee "$SES/manifest.txt"
fi
echo "engine: $(md5sum "$ENGINE" | cut -d' ' -f1)  threads=$THR len=$LEN ctx=$CTX chunk=$CHUNK"
VOCAB=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("vocab_size",0))' "$CONT/config.json" 2>/dev/null || echo 0)
echo "vocab: $VOCAB (E17 degenerate-logits tripwire reference)"

if [ ! -f "$SES/ref.json" ]; then
  # E15: the ref builder spawns the engine (tokenize + DUMP_IDS). It must run
  # the streaming dialect: TRUNK_MAP=1 keeps the F16 trunk mmap'd on disk (the
  # stock loader mallocs all 54.7 GB → dead in seconds on the 7.62 GB box), and
  # HOT_FRAC keeps the prefill inside RAM. The ids are pure tokenizer output —
  # generate() copies the prompt ids verbatim before any forward — so the frac
  # cannot taint ref.json. Right-sized slice (~6 bytes/token over ctx+len+128)
  # bounds the one-off prefill; --len=CTX+LEN so exactly LEN tokens get scored
  # (the old --len=LEN silently scored LEN−CTX).
  REF_BYTES=$(( (CTX + LEN + 128) * 6 )); [ "$REF_BYTES" -lt 2400 ] && REF_BYTES=2400
  env TRUNK_MAP=1 HOT_FRAC="${PHASE5_REF_FRAC:-0.25}" \
    python3 "$(dirname "$0")/../scripts/make_wikitest_ref.py" "$CONT" "$WIKI" "$SES/ref.json" \
    --ctx=$CTX --len=$((CTX + LEN)) --bytes=$REF_BYTES || exit 2
fi

# E17 tripwire: ppl == vocab_size to within 1e-4 relative IS the map_f16_one
# bias signature (mapped F16 weights at 2^-112 of true value -> logits ~1e-33
# -> softmax uniform to the last float -> NLL == ln(V) exactly). Every arm on
# such an engine measures a dead trunk; abort before the disk burns for hours.
ppl_degenerate() {  # ppl_degenerate <arm> -> rc 0 when degenerate
  local out="$SES/arms/$1.out" ppl
  [ -s "$out" ] || return 1
  [ "${VOCAB:-0}" -gt 0 ] 2>/dev/null || return 1
  ppl=$(sed -n 's/.*ppl = \([0-9.eE+-]*\).*/\1/p' "$out" | head -1)
  [ -n "$ppl" ] || return 1
  python3 -c 'import sys; ppl, v = float(sys.argv[1]), float(sys.argv[2]); sys.exit(0 if abs(ppl - v) <= 1e-4 * v else 1)' "$ppl" "$VOCAB"
}
e17_abort() {
  echo "!! arm $1: ppl == vocab_size ($VOCAB) — uniform logits, numerically dead trunk."
  echo "!! This is the E17 signature: the engine predates tools/patch_mapf16_bias.py"
  echo "!! (every mapped F16 weight reads 2^-112 of its true value; the whole trunk"
  echo "!! leaves the computation; tf-NLL comes out exactly ln(V))."
  echo "!! Fix, in the repo clone:  git pull"
  echo "!!   cp engine/qwen36.c engine/*.h ~/colibri/c/ && make -C ~/colibri/c qwen36 -j4"
  echo "!!   rm -f $SES/arms/*        # keep manifest.txt + ref.json (engine-independent)"
  echo "!! then rerun this exact session command."
  exit 3
}
arm() {  # arm <name> <extra env...>
  local name=$1; shift
  if [ -f "$SES/arms/$name.err" ] && [ "${PHASE5_FORCE:-0}" != 1 ]; then
    echo "== arm $name: cached"
    ppl_degenerate "$name" && e17_abort "$name"
    return
  fi
  echo "== arm $name: $*  (started $(date +%H:%M:%S))"
  env $COMMON "$@" "$ENGINE" "$CAP" "$THR" "$SES/ref.json" > "$SES/arms/$name.out" 2> "$SES/arms/$name.err"
  echo "rc=$?" >> "$SES/arms/$name.err"
  echo "== arm $name: done $(date +%H:%M:%S)"
  ppl_degenerate "$name" && e17_abort "$name"
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
echo "== M1: determinism (arm-A config, GENERATE mode via COMMON_GEN — PPL=1 would take tf_nll and return before generate(), so DUMP_IDS would never be written: E15b)"
M1NEW="${PHASE5_M1NEW:-1}"
if [ -s "$SES/arms/A.ids" ] && [ -s "$SES/arms/A2.ids" ] && [ "${PHASE5_FORCE:-0}" != 1 ]; then
  echo "== M1: cached id streams (PHASE5_FORCE=1 to rerun — costs 2× frac-1.0 passes in the field)"
else
  DUMP_IDS="$SES/arms/A.ids" env $COMMON_GEN HOT_FRAC=1.0 GRIFFIN=0 N_NEW="$M1NEW" "$ENGINE" "$CAP" "$THR" "$SES/ref.json" > /dev/null 2> "$SES/arms/A_det.err" || true
  DUMP_IDS="$SES/arms/A2.ids" env $COMMON_GEN HOT_FRAC=1.0 GRIFFIN=0 N_NEW="$M1NEW" "$ENGINE" "$CAP" "$THR" "$SES/ref.json" > /dev/null 2> "$SES/arms/A_det2.err" || true
fi
cmp -s "$SES/arms/A.ids" "$SES/arms/A2.ids" && echo "M1 PASS: identical id streams (prefill + $M1NEW greedy steps)" || echo "M1 FAIL: id streams differ"

python3 - "$SES" "$CONT" "$LEN" "$CHUNK" "$CTX" <<'PY'
import json, os, re, sys, time
ses, cont = sys.argv[1], sys.argv[2]
slen, schunk, sctx = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
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
    ppl = re.search(r"ppl = ([\d.eE+-]+)", txt)
    sc = re.search(r"over (\d+) tokens", txt)
    rss = re.search(r"PEAK RSS:\s*([\d.]+) GB", txt)
    rc = re.search(r"rc=(\d+)\s*$", txt)
    arms[name] = {"ppl": float(ppl.group(1)) if ppl else None,
                  "scored": int(sc.group(1)) if sc else None,
                  "peak_rss_gb": float(rss.group(1)) if rss else None,
                  "rc": int(rc.group(1)) if rc else None}
base = (arms.get("A") or {}).get("ppl")
for name, d in arms.items():
    d["delta_rel_vs_A"] = round((d["ppl"] - base) / base, 4) if (d["ppl"] and base) else None
    band = BANDS.get(name)
    d["verdict"] = ("PASS" if band and d["delta_rel_vs_A"] is not None and abs(d["delta_rel_vs_A"]) <= band
                    else ("n/a" if name in ("A",) or band is None else "FAIL")) if d["ppl"] else "no-data"
doc = {"session": ses, "container": cont, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "ctx": sctx, "scored_tokens": slen, "tf_chunk": schunk, "arms": arms,
       "note": "bands mirrored from docs/VALIDATION_PROTOCOL_QWEN36.md §3; D/E judged vs C in the report; "
               "E16: scored with Q36_TF_CHUNK (chunked tf-nll, CI-parity-gated vs per-token) — "
               "all arms share the chunk, so deltas stay paired"}
json.dump(doc, open(os.path.join(ses, "summary.json"), "w"), indent=1)
print("\n== session summary (%s/summary.json)" % ses)
print("arm  ppl       dRel(A)   scored  RSS GB  rc  verdict")
for name, d in arms.items():
    print("%-4s %-9s %-9s %-7s %-7s %-3s %s" % (name, d["ppl"], d["delta_rel_vs_A"], d["scored"], d["peak_rss_gb"], d["rc"], d["verdict"]))
print("\nnext: commit %s (or tar it) and paste the summary — the report doc gets written from these numbers" % ses)
PY
echo "== session complete: $SES"
