#!/bin/bash
# install_bundle.sh — install the q36 launcher, log tool and engine from an unpacked bundle, WITHOUT git.
# (tools/make_bundle.sh builds the bundle and ships this file inside it as install.sh.)
#
#   bash install.sh          run from the unpacked bundle directory
#   BIN=... CDIR=... bash install.sh   override ~/bin and ~/colibri/c
#
# 1. verifies the bundle (md5), 2. backs up everything it replaces, 3. copies, 4. rebuilds the engine,
# 5. verifies the result (checksums, and that the rebuilt binary contains every Q36_* knob the bundled
# source reads), and restores the backup if anything fails. Safe to run twice.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$HOME/bin}"; CDIR="${CDIR:-$HOME/colibri/c}"
TS="$(date +%Y%m%d-%H%M%S)"; BK="$HOME/q36-backup-$TS"

restore() {
  [ -d "$BK" ] || return 0
  echo "restoring from $BK ..."
  [ -f "$BK/q36-minipc" ]      && cp -a "$BK/q36-minipc" "$BIN/q36-minipc"
  [ -f "$BK/q36_benchlog.py" ] && cp -a "$BK/q36_benchlog.py" "$BIN/q36_benchlog.py"
  if [ -d "$BK/c" ]; then
    cp -a "$BK/c/." "$CDIR/"
    [ -f "$BK/c/qwen36.bin.old" ] && cp -a "$BK/c/qwen36.bin.old" "$CDIR/qwen36" && rm -f "$CDIR/qwen36.bin.old"
  fi
}
fail() { echo; echo "*** INSTALL FAILED: $* ***"; restore; exit 1; }

echo "== 1. verify the bundle"
( cd "$HERE" && md5sum -c MD5SUMS --quiet ) || { echo "bundle is corrupt or incomplete - re-download it"; exit 1; }
[ -d "$CDIR" ] || { echo "colibri directory not found: $CDIR (set CDIR=...)"; exit 1; }
mkdir -p "$BIN"

echo "== 2. back up what will be replaced -> $BK"
mkdir -p "$BK/c"
[ -f "$BIN/q36-minipc" ]      && cp -a "$BIN/q36-minipc" "$BK/"
[ -f "$BIN/q36_benchlog.py" ] && cp -a "$BIN/q36_benchlog.py" "$BK/"
[ -f "$CDIR/qwen36.c" ]       && cp -a "$CDIR/qwen36.c" "$BK/c/"
for h in "$HERE"/engine/*.h; do
  [ -f "$CDIR/$(basename "$h")" ] && cp -a "$CDIR/$(basename "$h")" "$BK/c/"
done
[ -f "$CDIR/qwen36" ] && cp -a "$CDIR/qwen36" "$BK/c/qwen36.bin.old"

echo "== 3. install"
cp "$HERE/tools/q36-minipc.sh" "$BIN/q36-minipc" && chmod +x "$BIN/q36-minipc" || fail "copying the launcher"
cp "$HERE/tools/q36_benchlog.py" "$BIN/q36_benchlog.py" || fail "copying the log tool"
cp "$HERE/engine/qwen36.c" "$CDIR/qwen36.c" && cp "$HERE"/engine/*.h "$CDIR/" || fail "copying the engine sources"

echo "== 4. rebuild the engine"
make -C "$CDIR" qwen36 || fail "make failed"

echo "== 5. verify"
want() { grep " $1\$" "$HERE/MD5SUMS" | cut -d' ' -f1; }
chk() { [ "$(md5sum "$1" | cut -d' ' -f1)" = "$2" ] || fail "$1 does not match the bundle"; }
chk "$BIN/q36-minipc"      "$(want tools/q36-minipc.sh)"
chk "$BIN/q36_benchlog.py" "$(want tools/q36_benchlog.py)"
chk "$CDIR/qwen36.c"       "$(want engine/qwen36.c)"
[ -n "$(find "$CDIR/qwen36" -newer "$HERE/MD5SUMS" 2>/dev/null)" ] || fail "the engine binary was not rebuilt"
n=0
for k in $(grep -o 'getenv("Q36_[A-Z0-9_]*")' "$HERE/engine/qwen36.c" | sed 's/getenv("//;s/")//' | sort -u); do
  grep -q "$k" "$CDIR/qwen36" || fail "the rebuilt engine binary does not contain $k (stale build?)"
  n=$((n + 1))
done
echo "all $n Q36_* knobs read by the bundled source are present in the rebuilt binary"
echo
md5sum "$BIN/q36-minipc" "$BIN/q36_benchlog.py" "$CDIR/qwen36.c" "$CDIR/qwen36"
echo
echo "UPDATED OK   (backup of the previous files: $BK)"
