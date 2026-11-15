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

# Isolate from any installed copy of the tools ($HOME/bin is searched first).
export HOME="$WORK/home"; mkdir -p "$HOME"
unset Q36_KVQ Q36_LEVEL Q36_EMBED_MAP Q36_EMBED_STREAMED Q36_ENGINE

# A dense container: qwen36_meta.json with num_experts=0 is all the launcher + projector read.
MODEL="$WORK/dense"; mkdir -p "$MODEL"
echo '{"hidden":5120,"shared_inter":17408,"n_layers":64,"num_experts":0}' > "$MODEL/qwen36_meta.json"

run() {  # run <outfile> <env...> -- <args...>  (stdout+stderr -> outfile, returns the rc)
  local out="$1"; shift
  env "$@" > "$out" 2>&1
}

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

[ "$fail" -eq 0 ] && echo "launcher gate: PASS" || { echo "launcher gate: FAIL"; exit 1; }
