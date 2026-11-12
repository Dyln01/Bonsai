#!/usr/bin/env bash
# check_embed_map.sh — Q36_EMBED_MAP gate: the mapped table must be IDENTICAL and ENGAGED.
#
# The patch stops holding `model.embed_tokens.weight` resident (vocab*hidden: 2.54 GB f16,
# 5.09 GB f32 at the 27B geometry) and reads the rows the forward needs through
# st_read_slice_f32. Two things must be true before that is shippable, and neither is
# visible from a timing:
#
#   1. IDENTITY — the id stream must be byte-for-byte the resident path's. A row read off
#      the wrong offset still generates fluent-looking text; only the id stream catches it.
#   2. ENGAGEMENT — the mapped path must actually run. An idle table and a table that was
#      never touched look the same in a timing, so the engine prints the row count
#      ("[embed] mapped: N row read(s)") and this gate requires N > 0.
#
# The RAM win itself is arithmetic on the real geometry (V*D*esz) and is NOT measurable on
# the CI container, whose table is ~4 MB by design; RSS is reported, not asserted.
#
# Usage: bash scripts/check_embed_map.sh [<engine-binary>] [<container>]
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENG="${1:-$HOME/colibri/c/qwen36}"
SNAP="${2:-$HOME/synth_models/small_dense_f16}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

if [ ! -x "$ENG" ]; then echo "FAIL  no engine binary at $ENG"; exit 1; fi
if [ ! -f "$SNAP/qwen36_meta.json" ]; then
  echo "note: no container at $SNAP -- building the small dense fixture"
  python3 "$HERE/scripts/gen_small_dense_f16.py" >/dev/null
fi
printf 'The quick brown fox jumps over the lazy dog and keeps running to the end. ' > "$OUT/prompt.txt"

run() {  # run <tag> <Q36_EMBED_MAP>
  env SNAP="$SNAP" Q36_EMBED_MAP="$2" DUMP_IDS="$OUT/ids_$1.txt" N_NEW=16 COLI_TEMP=0 \
      OMP_NUM_THREADS=2 COLI_NO_OMP_TUNE=1 KVSAVE=0 \
      /usr/bin/time -v "$ENG" 4 2 "$OUT/prompt.txt" \
      > "$OUT/$1.out" 2> "$OUT/$1.err"
  echo $? > "$OUT/$1.rc"
  grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' "$OUT/$1.err" | grep -oE '[0-9]+$'
}

rss_res="$(run resident 0)"; rc_res="$(cat "$OUT/resident.rc")"
rss_map="$(run mapped   1)"; rc_map="$(cat "$OUT/mapped.rc")"

echo "arm resident: rc=$rc_res maxRSS=${rss_res:-?} kB"
echo "arm mapped  : rc=$rc_map maxRSS=${rss_map:-?} kB"
grep -E '^\[embed\]' "$OUT/mapped.err" | sed 's/^/  /' || true

[ "$rc_res" = "0" ] && [ "$rc_map" = "0" ] || { echo "FAIL  an arm exited non-zero (resident=$rc_res mapped=$rc_map)"; fail=1; }

if cmp -s "$OUT/ids_resident.txt" "$OUT/ids_mapped.txt" && [ -s "$OUT/ids_resident.txt" ]; then
  echo "PASS  id-identity: mapped == resident ($(md5sum < "$OUT/ids_resident.txt" | cut -c1-12))"
else
  echo "FAIL  id-identity: the mapped row read changed the id stream"
  fail=1
fi

rows="$(grep -oE 'mapped: [0-9]+ row read' "$OUT/mapped.err" | grep -oE '[0-9]+' | head -1)"
if [ -n "${rows:-}" ] && [ "$rows" -gt 0 ]; then
  echo "PASS  mapped path engaged ($rows row read(s) off disk)"
else
  echo "FAIL  the mapped path never ran (no row count printed)"
  fail=1
fi

if [ -n "${rss_res:-}" ] && [ -n "${rss_map:-}" ]; then
  echo "note  RSS ${rss_res} -> ${rss_map} kB on a ~4 MB synthetic table (the real win is"
  echo "      V*D*esz: 2.54 GB f16 / 5.09 GB f32 at the 27B geometry, priced by project --floor)"
fi

[ "$fail" -eq 0 ] && echo "== check_embed_map: ALL PASS ==" || echo "== check_embed_map: FAILURES =="
exit $fail
