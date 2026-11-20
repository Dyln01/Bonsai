#!/usr/bin/env bash
# check_bundle.sh — gate for tools/make_bundle.sh + tools/install_bundle.sh (the git-free installer).
# Simulates a box (old launcher/log tool/engine, a fake `make` that copies the source to the binary) and checks:
#   1. a good bundle installs, verifies, and reports UPDATED OK        4. a corrupt bundle changes NOTHING
#   2. a second run is harmless                                        5. a stale build (binary not rebuilt) is caught
#   3. a failing build restores the previous files
# usage: bash scripts/check_bundle.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
fail=0
ok()  { echo "ok    $*"; }
bad() { echo "FAIL  $*"; fail=1; }

bash "$HERE/tools/make_bundle.sh" "$W/out" >/dev/null || { echo "FAIL  make_bundle.sh failed"; exit 1; }
tar xzf "$W/out/q36_bundle.tar.gz" -C "$W"
BUNDLE="$W/q36_bundle"

reset_box() {   # an "old" installation
  rm -rf "$W/box"; mkdir -p "$W/box/home/bin" "$W/box/c"
  cp "$HERE/engine/qwen36.c.premapf16" "$W/box/c/qwen36.c"
  echo OLD-LAUNCHER > "$W/box/home/bin/q36-minipc"; echo OLD-LOGTOOL > "$W/box/home/bin/q36_benchlog.py"
  printf 'qwen36: qwen36.c\n\tcp qwen36.c qwen36\n' > "$W/box/c/Makefile"
  ( cd "$W/box/c" && make qwen36 >/dev/null 2>&1 )
  touch -d "2020-01-01" "$W/box/c/qwen36"
}
install() { HOME="$W/box/home" BIN="$W/box/home/bin" CDIR="$W/box/c" bash "${1:-$BUNDLE}/install.sh" 2>&1; }
md5() { md5sum "$1" | cut -d' ' -f1; }

reset_box; out="$(install)"; rc=$?
[ "$rc" -eq 0 ] && echo "$out" | grep -q "UPDATED OK" && ok "good bundle installs (UPDATED OK)" || bad "good bundle did not install: $(echo "$out" | tail -2 | tr '\n' ' ')"
[ "$(md5 "$W/box/home/bin/q36-minipc")" = "$(md5 "$HERE/tools/q36-minipc.sh")" ] && ok "installed launcher == repo launcher" || bad "installed launcher differs from the repo's"
[ "$(md5 "$W/box/home/bin/q36_profile.sh")" = "$(md5 "$HERE/tools/q36_profile.sh")" ] && ok "installed profiler == repo profiler" || bad "installed profiler differs from the repo's"
[ "$(md5 "$W/box/c/qwen36.c")" = "$(md5 "$HERE/engine/qwen36.c")" ] && ok "installed engine source == repo engine source" || bad "installed engine source differs"
ls "$W/box/home"/q36-backup-*/q36-minipc >/dev/null 2>&1 && grep -q OLD-LAUNCHER "$W/box/home"/q36-backup-*/q36-minipc && ok "previous launcher was backed up" || bad "no backup of the previous launcher"

out="$(install)"; echo "$out" | grep -q "UPDATED OK" && ok "second run is harmless" || bad "second run failed"

reset_box; printf 'qwen36: qwen36.c\n\tfalse\n' > "$W/box/c/Makefile"
out="$(install)"; rc=$?
[ "$rc" -ne 0 ] && echo "$out" | grep -q "INSTALL FAILED" && ok "failing build is reported (exit $rc)" || bad "failing build not reported"
[ "$(cat "$W/box/home/bin/q36-minipc")" = OLD-LAUNCHER ] && cmp -s "$W/box/c/qwen36.c" "$HERE/engine/qwen36.c.premapf16" && ok "failing build restored the old launcher and engine source" || bad "failing build did NOT restore the previous files"

reset_box; rm -rf "$W/bad"; cp -r "$BUNDLE" "$W/bad"; echo garbage >> "$W/bad/engine/qwen36.c"
out="$(install "$W/bad")"; rc=$?
# rejected UP FRONT: no backup directory (step 2) and nothing copied or rebuilt. A rollback after the fact would
# leave the same files behind, so the absence of a backup dir is what proves the checksum check ran first.
[ "$rc" -ne 0 ] && [ "$(cat "$W/box/home/bin/q36-minipc")" = OLD-LAUNCHER ] && cmp -s "$W/box/c/qwen36.c" "$HERE/engine/qwen36.c.premapf16" \
  && ! ls "$W/box/home"/q36-backup-* >/dev/null 2>&1 && ok "corrupt bundle rejected up front (no backup made, nothing touched)" || bad "corrupt bundle was not rejected up front"

reset_box; printf 'qwen36: qwen36.c\n\ttrue\n' > "$W/box/c/Makefile"
out="$(install)"; rc=$?
[ "$rc" -ne 0 ] && echo "$out" | grep -qE "not rebuilt|stale build" && ok "stale build (binary not rebuilt) is caught" || bad "stale build was NOT caught"

[ "$fail" -eq 0 ] && echo "bundle gate: PASS" || { echo "bundle gate: FAIL"; exit 1; }
