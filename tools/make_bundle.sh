#!/bin/bash
# make_bundle.sh — build a git-free install bundle from this checkout.
#   bash tools/make_bundle.sh [outdir]      -> <outdir>/q36_bundle.tar.gz   (default outdir: .)
# Unpack on the box and run:  bash q36_bundle/install.sh   (see tools/install_bundle.sh)
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-.}"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
B="$W/q36_bundle"; mkdir -p "$B/engine" "$B/tools"
cp "$ROOT/tools/q36-minipc.sh" "$ROOT/tools/q36_benchlog.py" "$ROOT/tools/q36_profile.sh" "$ROOT/tools/quality_probe_text.txt" "$ROOT/scripts/make_wikitest_ref.py" "$B/tools/"
cp "$ROOT/engine/qwen36.c" "$ROOT"/engine/*.h "$B/engine/"
cp "$ROOT/tools/install_bundle.sh" "$B/install.sh"; chmod +x "$B/install.sh"
( cd "$B" && md5sum tools/* engine/* > MD5SUMS )
( cd "$W" && tar czf "$OUT/q36_bundle.tar.gz" q36_bundle )
echo "built $OUT/q36_bundle.tar.gz ($(git -C "$ROOT" log -1 --format='%h %s' 2>/dev/null | cut -c1-60))"
