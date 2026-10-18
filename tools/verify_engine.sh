#!/usr/bin/env bash
# verify_engine.sh — runnable gates over the engine/ snapshot chain.
#
#   1. every file in engine/ matches tools/patch_manifest.json (md5)
#   2. the reproducible patch segments replay exactly:
#        premap -trunkmap-> -trunkmap_v2-> prehot
#        pretq  -kvtq-> presm
#        presm  -softmax-> qwen36.c
#   3. if a colibri checkout exists (COLIBI_DIR or ~/colibri):
#        qwen36.c.orig == bf24429 c/qwen36.c (stock)
#
# Segments orig->premap and prehot->pretq are NOT replayable with the committed
# patchers (drift audit: tools/patch_manifest.json, docs/ROADMAP.md item E1);
# until that audit closes, the engine/ snapshots + this manifest are the
# source of truth, and this script fails loudly if any of them moves.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0

python3 - "$ROOT" <<'PY' || fail=1
import json, hashlib, sys, os
root = sys.argv[1]
man = json.load(open(os.path.join(root, "tools", "patch_manifest.json")))
ok = True
for name, md in man["engine_files"].items():
    p = os.path.join(root, "engine", name)
    got = hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    if got != md:
        print("FAIL  md5 engine/%s (%s)" % (name, got)); ok = False
print("PASS  engine snapshot md5s (%d files)" % len(man["engine_files"]) if ok
      else "FAIL  engine snapshot md5s")
sys.exit(0 if ok else 1)
PY

step() { # step <from> <to|FINAL> <patch.py...>
  local from=$1 to=$2; shift 2
  cp "$ROOT/engine/$from" "$T/qwen36.c"
  local p
  for p in "$@"; do
    python3 "$ROOT/tools/$p" "$T/qwen36.c" >/dev/null 2>&1 || { echo "FAIL  $from: $p errored"; fail=1; return; }
  done
  local want="$ROOT/engine/$to"; [ "$to" = FINAL ] && want="$ROOT/engine/qwen36.c"
  if cmp -s "$T/qwen36.c" "$want"; then echo "PASS  $from -> $(basename "$want") [$*]"
  else echo "FAIL  $from -> $(basename "$want") [$*]"; fail=1; fi
}
step qwen36.c.premap qwen36.c.prehot patch_trunkmap.py patch_trunkmap_v2.py
step qwen36.c.pretq  qwen36.c.presm  patch_kvtq.py
step qwen36.c.presm  FINAL           patch_softmax.py

COLIBI="${COLIBI_DIR:-$HOME/colibri}"
if [ -d "$COLIBI/.git" ]; then
  s=$(git -C "$COLIBI" show bf24429:c/qwen36.c 2>/dev/null | md5sum | cut -d' ' -f1)
  o=$(md5sum "$ROOT/engine/qwen36.c.orig" | cut -d' ' -f1)
  if [ "$s" = "$o" ]; then echo "PASS  qwen36.c.orig == colibri bf24429 stock"
  else echo "FAIL  qwen36.c.orig vs stock ($s != $o)"; fail=1; fi
else
  echo "SKIP  stock check (no colibri checkout at $COLIBI; set COLIBI_DIR to enable)"
fi

if [ "$fail" -eq 0 ]; then echo "== verify_engine: ALL PASS =="; else echo "== verify_engine: FAILURES =="; fi
exit $fail
