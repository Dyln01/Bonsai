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
# Segments replay the FULL chain since the E1 audit closed (2026-10-19):
#   orig -kv8_flash_v1-> premap -trunkmap-> -trunkmap_v2-> prehot
#        -hotstack_v1-> pretq -kvtq-> presm -softmax-> premtp -mtp->
#        -mtp_draft-> -mtp_oracle-> -mtp_moe-> prei4 -mtp_i4-> pregqa
#        -gqa_absorb-> prebar -calib_bar-> preabsorb -tq_absorb->
#        preabsp -tq_absorb_prefill-> premapsc -mapsc-> prescan
#        -scan_decay-> preptk -profile_tok-> qwen36.c
# The historical per-feature patchers for the two regenerated windows live in
# tools/attic/ (see tools/attic/ATTIC.md).
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
step qwen36.c.orig     qwen36.c.premap  patch_kv8_flash_v1.py
step qwen36.c.premap   qwen36.c.prehot  patch_trunkmap.py patch_trunkmap_v2.py
step qwen36.c.prehot   qwen36.c.pretq   patch_hotstack_v1.py
step qwen36.c.pretq    qwen36.c.presm   patch_kvtq.py
step qwen36.c.presm  qwen36.c.premtp patch_softmax.py
step qwen36.c.premtp qwen36.c.predraft patch_mtp.py
step qwen36.c.predraft qwen36.c.preoracle patch_mtp_draft.py
step qwen36.c.preoracle qwen36.c.premoe patch_mtp_oracle.py
step qwen36.c.premoe   qwen36.c.prei4 patch_mtp_moe.py
step qwen36.c.prei4    qwen36.c.pregqa patch_mtp_i4.py
step qwen36.c.pregqa   qwen36.c.prebar patch_gqa_absorb.py
step qwen36.c.prebar   qwen36.c.preabsorb patch_calib_bar.py
step qwen36.c.preabsorb qwen36.c.preabsp patch_tq_absorb.py
step qwen36.c.preabsp  qwen36.c.premapsc patch_tq_absorb_prefill.py
step qwen36.c.premapsc qwen36.c.prescan patch_mapsc.py
step qwen36.c.prescan  qwen36.c.preptk patch_scan_decay.py
step qwen36.c.preptk   qwen36.c.preftz patch_profile_tok.py
step qwen36.c.preftz   qwen36.c.prestream patch_ftz.py
step qwen36.c.prestream FINAL            patch_trunk_stream.py

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
