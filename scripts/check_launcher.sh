#!/usr/bin/env bash
# check_launcher.sh — smoke test for tools/q36-minipc.sh (the one-knob launcher).
#
# Why this exists: the launcher runs under `set -u` and composes its engine environment in
# one place for every dense-model command. A single misspelled variable ($EMMAP for
# $EMBMAP) therefore killed `floor`, `project`, `bench`, ... before dispatch, and nothing
# in CI noticed because no gate executed the launcher. This gate runs it, hermetically
# (no engine binary, no model weights, no network): the launcher is pointed at a
# meta-only dense container and the commands that need no engine are exercised.
#
#   1. SYNTAX      bash -n, plus shellcheck's misspelled/unassigned-variable checks
#                  (SC2153/SC2154) when shellcheck is installed.
#   2. HELP        exits 0 and lists EVERY command the dispatch `case` handles, so a new
#                  command cannot ship undocumented (the way `floor` did).
#   3. NO ABORT    help / project / floor on a dense container exit 0 and never print
#                  "unbound variable", with and without the optional knobs set.
#   4. KV DEFAULT  `floor` models the launcher's resolved KV tier: no Q36_KVQ == kv8, and
#                  both differ from an explicit f32 (the projector's own default).
#   5. BAD INPUT   an unknown Q36_KVQ is rejected (exit 2), not silently ignored.
#
# Usage: bash scripts/check_launcher.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHER="$HERE/tools/q36-minipc.sh"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
fail=0

ok()  { echo "ok    $*"; }
bad() { echo "FAIL  $*"; fail=1; }

# Isolate from any installed copy of the tools ($HOME/bin is searched first). Python's user
# site (where `pip install --user`/PEP 668 systems put numpy) lives under the REAL $HOME, so
# pin it before HOME moves, or numpy "disappears" on any runner that does not install it
# system-wide.
PYTHONUSERBASE="${PYTHONUSERBASE:-$(python3 -m site --user-base 2>/dev/null)}"; export PYTHONUSERBASE
export HOME="$WORK/home"; mkdir -p "$HOME"
unset Q36_KVQ Q36_LEVEL Q36_EMBED_MAP Q36_EMBED_STREAMED Q36_ENGINE

# A dense container: qwen36_meta.json with num_experts=0 is all the launcher + projector read.
MODEL="$WORK/dense"; mkdir -p "$MODEL"
echo '{"hidden":5120,"shared_inter":17408,"n_layers":64,"num_experts":0}' > "$MODEL/qwen36_meta.json"

run() {  # run <outfile> <env...> -- <args...>  (stdout+stderr -> outfile, returns the rc)
  local out="$1"; shift
  env "$@" > "$out" 2>&1
}

# ---- 0. prerequisites (fail ONCE and clearly, not as a wall of cryptic per-command errors) ----
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "FAIL  python3 cannot import numpy -- the projector behind floor/project needs it (pip install numpy)"
  echo "launcher gate: FAIL"; exit 1
fi

# ---- 1. syntax ----
if bash -n "$LAUNCHER"; then ok "bash -n"; else bad "bash -n: syntax error in $LAUNCHER"; fi
if command -v shellcheck >/dev/null 2>&1; then
  if sc="$(shellcheck -s bash -i SC2153,SC2154 "$LAUNCHER" 2>&1)"; then
    ok "shellcheck SC2153/SC2154 (misspelled / unassigned variables)"
  else
    bad "shellcheck found a misspelled or unassigned variable:"; echo "$sc" | sed 's/^/        /'
  fi
else
  echo "note  shellcheck not installed -- skipping the static variable check"
fi

# ---- 2. help lists every dispatched command ----
run "$WORK/help.out" bash "$LAUNCHER" help; rc=$?
[ "$rc" -eq 0 ] && ok "help exits 0" || bad "help exited $rc"
# the dispatch is the LAST top-level `case "$CMD" in` ... `esac`; its arms are 2-space-indented
cmds="$(awk '
  { line[NR]=$0 }
  END {
    for (i=1;i<=NR;i++) if (line[i] ~ /^case "\$CMD" in$/) start=i
    for (i=start+1;i<=NR && line[i] !~ /^esac$/;i++)
      if (match(line[i], /^  [a-z][a-z|]*\)/)) {
        m=split(substr(line[i], 3, RLENGTH-3), p, "|")
        for (j=1;j<=m;j++) if (p[j] != "help") print p[j]
      }
  }' "$LAUNCHER")"
[ -n "$cmds" ] || bad "could not extract the dispatched command list (did the case layout change?)"
missing=""
for c in $cmds; do grep -q "q36-minipc $c " "$WORK/help.out" || missing="$missing $c"; done
if [ -z "$missing" ]; then ok "help documents every command ($(echo $cmds | wc -w) checked)"
else bad "commands handled but absent from help:$missing"; fi

# ---- 3. no abort on a dense container ----
for sub in project floor; do
  for knobs in "" "Q36_EMBED_MAP=1" "Q36_EMBED_STREAMED=1 Q36_KVQ=tq4"; do
    label="$sub ${knobs:+[$knobs]}"
    # shellcheck disable=SC2086
    run "$WORK/o.out" $knobs bash "$LAUNCHER" "$sub" "$MODEL"; rc=$?
    if [ "$rc" -ne 0 ]; then bad "$label exited $rc: $(grep -v '^$' "$WORK/o.out" | tail -1)"
    elif grep -qi "unbound variable" "$WORK/o.out"; then bad "$label: unbound variable"
    else ok "$label"; fi
  done
done
run "$WORK/floor.out" bash "$LAUNCHER" floor "$MODEL"
grep -q "physical floor" "$WORK/floor.out" && ok "floor prints its report" || bad "floor produced no report"

# ---- 4. floor's KV default is the launcher's (kv8), not the projector's (f32) ----
kvline() { grep -m1 "cache_room" "$1"; }
run "$WORK/k_def.out" bash "$LAUNCHER" floor "$MODEL"
run "$WORK/k_kv8.out" Q36_KVQ=kv8 bash "$LAUNCHER" floor "$MODEL"
run "$WORK/k_f32.out" Q36_KVQ=f32 bash "$LAUNCHER" floor "$MODEL"
d="$(kvline "$WORK/k_def.out")"; k8="$(kvline "$WORK/k_kv8.out")"; f="$(kvline "$WORK/k_f32.out")"
if [ -z "$d" ] || [ -z "$k8" ] || [ -z "$f" ]; then bad "floor: no cache_room line to compare"
else
  [ "$d" = "$k8" ] && ok "floor default == Q36_KVQ=kv8" || bad "floor default is not kv8: [$d] vs [$k8]"
  [ "$d" != "$f" ]  && ok "floor default != explicit f32 (the tiers really differ)" \
                    || bad "floor ignores the KV tier: default and f32 give the same cache_room"
fi

# ---- 5. bad input is rejected ----
run "$WORK/bad.out" Q36_KVQ=bogus bash "$LAUNCHER" floor "$MODEL"; rc=$?
[ "$rc" -eq 2 ] && ok "unknown Q36_KVQ rejected (exit 2)" || bad "unknown Q36_KVQ gave exit $rc (want 2)"

# ---- 6. bench writes a complete log (fake engine; needs GNU time like the real bench) ----
if [ ! -x /usr/bin/time ]; then
  echo "SKIP  bench-log checks: GNU time not installed (apt install time) -- CI installs it"
else
  FAKE="$WORK/fake_engine"
  cat > "$FAKE" <<'EOS'
#!/bin/bash
echo "== qwen36 Phase-2 engine | cache=48/layer bits=4 ctx=8192 hot=0 ==" >&2
echo "resident weights loaded in 1.0s | RSS after load: 0.01 GB" >&2
[ "${Q36_WIN_MB:-0}" != 0 ] && echo "[qwen36 win] hits 5 misses 10 (33.3% hit-rate), budget ${Q36_WIN_MB} MB" >&2
sleep 3
[ -n "${FAKE_FAIL:-}" ] && { echo "engine exploded" >&2; exit 3; }
echo "TTFT: 1.50 s (time to first token)" >&2
echo "[timers] decode: 3 tokens  (shared/router are subsets of moe_total)" >&2
echo "[timers]   TOTAL      900.0 ms  300.00 ms/token" >&2
echo "PEAK RSS: 0.04 GB" >&2
echo "def is_palindrome(s): return s == s[::-1]"
EOS
  chmod +x "$FAKE"; mkdir -p "$WORK/logs"
  bench() { env Q36_ENGINE="$FAKE" Q36_LOG_DIR="$WORK/logs" Q36_SAMPLE_S=1 Q36_WIN_MB=0 Q36_GRIFFIN=0 "$@" bash "$LAUNCHER" bench "$MODEL"; }

  bench Q36_BENCH_TOKENS=3 > "$WORK/b1.out" 2>&1; rc=$?
  LOGF="$(ls -t "$WORK"/logs/bench_*.log 2>/dev/null | head -1)"
  [ "$rc" -eq 0 ] && ok "bench exits 0" || bad "bench exited $rc: $(tail -2 "$WORK/b1.out" | tr '\n' ' ')"
  if [ -n "$LOGF" ] && [ -s "$LOGF" ]; then
    ok "bench wrote $(basename "$LOGF")"
    for needle in "== SUMMARY ==" "DECODE        : 0.30 s/token" "N_NEW=3" "GRIFFIN=0" "ok engine environment matches" \
                  "== MACHINE ==" "== TIMELINE" "def is_palindrome" "== TOP MEMORY USERS" \
                  "Q36_NO_VISION=1" "ok swap:" "anonymous (non-mappable) memory"; do
      grep -qF -- "$needle" "$LOGF" && ok "log has: $needle" || bad "log is missing: $needle"
    done
    grep -q "KNOB MISMATCH" "$LOGF" && bad "log reports a knob mismatch on a clean run"
    grep -E "^command +: .*q36-minipc.* bench " "$LOGF" >/dev/null && ok "log records the command as typed" \
      || bad "log records the wrong command line: $(grep -m1 '^command' "$LOGF" | cut -c1-90)"
  else
    bad "bench wrote no log in $WORK/logs"
  fi

  FAKE_FAIL=1 bench Q36_BENCH_TOKENS=3 > "$WORK/b2.out" 2>&1; rc=$?
  [ "$rc" -eq 3 ] && ok "a failing engine makes bench exit with ITS status (3)" || bad "failing engine: bench exited $rc (want 3)"
  F2="$(ls -t "$WORK"/logs/bench_*.log 2>/dev/null | head -1)"
  grep -q "ENGINE FAILED OR WAS INTERRUPTED" "$F2" 2>/dev/null && ok "failed run is still logged and flagged" || bad "failed run not flagged in its log"

  bench Q36_BENCH_TOKENS=3 Q36_THREADS=3 > "$WORK/b4.out" 2>&1
  LOGT="$(ls -t "$WORK"/logs/bench_*.log 2>/dev/null | head -1)"
  grep -q "^  OMP_NUM_THREADS=3" "$LOGT" 2>/dev/null && ok "Q36_THREADS=3 reaches the engine as OMP_NUM_THREADS=3" || bad "Q36_THREADS did not reach the engine"
  bench Q36_BENCH_TOKENS=3 Q36_THREADS=zero > "$WORK/b5.out" 2>&1; rc=$?
  [ "$rc" -eq 2 ] && ok "bad Q36_THREADS rejected (exit 2)" || bad "bad Q36_THREADS gave exit $rc (want 2)"

  bench Q36_BENCH_TOKENS=abc > "$WORK/b3.out" 2>&1; rc=$?
  [ "$rc" -eq 2 ] && ok "bad Q36_BENCH_TOKENS rejected (exit 2)" || bad "bad Q36_BENCH_TOKENS gave exit $rc (want 2)"
fi

# ---- 7. quality: perplexity arms through the engine's PPL=1 contract (fake engine) ----
if [ ! -x /usr/bin/time ]; then
  echo "SKIP  quality checks: GNU time not installed"
else
  FAKEQ="$WORK/fake_q"
  cat > "$FAKEQ" <<'EOS'
#!/bin/bash
# honours the two contracts make_wikitest_ref.py and the quality arms rely on
if [ -n "${DUMP_IDS:-}" ]; then seq 1 900 > "$DUMP_IDS"; exit 0; fi          # reference build: ids of the text + 1 generated
if [ "${PPL:-}" = 1 ]; then
  echo "[tf] chunk=64: 128 scored tokens in 3 weight passes (0 per-token)" >&2
  echo "ARM HOT_FRAC=${HOT_FRAC:-?} DOWN_COMPACT=${Q36_DOWN_COMPACT:-unset} GRIFFIN=${GRIFFIN:-?} KV8=${KV8:-?}" >&2
  if [ -n "${FAKE_DEAD:-}" ]; then ppl=248320.00; elif [ "${HOT_FRAC:-}" = 1.0 ]; then ppl=6.74; else ppl=1791.28; fi
  echo "TF-NLL: 1.9076 nats/token over 128 tokens | ppl = $ppl"
  exit 0
fi
exit 1
EOS
  chmod +x "$FAKEQ"; mkdir -p "$WORK/qlogs"; rm -f /tmp/q36_quality_ref_n128.json
  quality() { env Q36_ENGINE="$FAKEQ" Q36_LOG_DIR="$WORK/qlogs" "$@" bash "$LAUNCHER" quality "$MODEL" 0.25 1.0; }

  quality > "$WORK/q1.out" 2>&1; rc=$?
  QLOGF="$(ls -t "$WORK"/qlogs/quality_*.log 2>/dev/null | head -1)"
  [ "$rc" -eq 0 ] && ok "quality exits 0" || bad "quality exited $rc: $(tail -2 "$WORK/q1.out" | tr '\n' ' ')"
  if [ -n "$QLOGF" ] && [ -s "$QLOGF" ]; then
    ok "quality wrote $(basename "$QLOGF")"
    grep -qE "^0\.25 +1\.9076 +1791\.28 .*GARBAGE" "$QLOGF" && ok "frac 0.25 at ppl 1791 is called GARBAGE vs the frac 1.0 reference" || bad "garbage verdict missing: $(grep -m1 '^0.25' "$QLOGF")"
    grep -qE "^1\.0 +1\.9076 +6\.74 .*reference" "$QLOGF" && ok "frac 1.0 is the reference" || bad "reference row missing"
    grep -q "ARM HOT_FRAC=0.25 DOWN_COMPACT=1 GRIFFIN=0 KV8=0" "$QLOGF" && ok "arm 0.25: GRIFFIN off, f32 KV, down-compact on" || bad "arm 0.25 settings wrong: $(grep -m1 'ARM HOT_FRAC=0.25' "$QLOGF")"
    grep -q "ARM HOT_FRAC=1.0 DOWN_COMPACT=unset" "$QLOGF" && ok "arm 1.0 does not compact the full-width down_proj" || bad "arm 1.0 got Q36_DOWN_COMPACT"
  else
    bad "quality wrote no log in $WORK/qlogs"
  fi
  quality FAKE_DEAD=1 > "$WORK/q2.out" 2>&1
  grep -q "DEAD TRUNK" "$WORK/q2.out" && ok "ppl == vocab size is flagged DEAD TRUNK" || bad "dead-trunk tripwire did not fire"
  env Q36_ENGINE="$FAKEQ" Q36_LOG_DIR="$WORK/qlogs" bash "$LAUNCHER" quality "$MODEL" abc > "$WORK/q3.out" 2>&1; rc=$?
  [ "$rc" -eq 2 ] && ok "non-numeric frac rejected (exit 2)" || bad "frac 'abc' gave exit $rc (want 2)"
  env Q36_ENGINE="$FAKEQ" Q36_LOG_DIR="$WORK/qlogs" bash "$LAUNCHER" quality "$MODEL" 1.5 > "$WORK/q4.out" 2>&1; rc=$?
  [ "$rc" -eq 2 ] && ok "frac above 1 rejected (exit 2)" || bad "frac 1.5 gave exit $rc (want 2)"
  mkdir -p "$WORK/solo" && cp "$LAUNCHER" "$WORK/solo/q36-minipc.sh"
  env Q36_ENGINE="$FAKEQ" Q36_LOG_DIR="$WORK/qlogs" bash "$WORK/solo/q36-minipc.sh" quality "$MODEL" 0.25 > "$WORK/q5.out" 2>&1; rc=$?
  [ "$rc" -eq 2 ] && grep -q "quality_probe_text.txt" "$WORK/q5.out" && ok "missing helper files give a clear error (exit 2)" || bad "missing helpers not reported (rc=$rc)"
fi

[ "$fail" -eq 0 ] && echo "launcher gate: PASS" || { echo "launcher gate: FAIL"; exit 1; }
