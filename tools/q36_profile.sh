#!/usr/bin/env bash
# q36_profile.sh — where does the engine's decode time actually go?  (sampling profiler, no rebuild)
#
#   bash tools/q36_profile.sh [seconds] [--now]
#
# Run it in a SECOND terminal while `q36-minipc bench ...` runs. It waits until the engine is in its
# steady compute-bound phase (its cores busy), then samples every thread's stack for
# <seconds> (default 40) and writes a ranked report to $Q36_LOG_DIR (default /home/dylan/sambashare)
# as profile_<time>.txt. Uses `perf` if it works, else `gdb` (apt install gdb). --now skips the wait.
# Sampling pauses the engine briefly, so the run is slower while profiled; the ranking is what matters.
set -u
SECS="${1:-40}"; [ "${SECS}" = "--now" ] && SECS=40
NOW=0; for a in "$@"; do [ "$a" = "--now" ] && NOW=1; done
OUTDIR="${Q36_LOG_DIR:-/home/dylan/sambashare}"
{ mkdir -p "$OUTDIR" && [ -w "$OUTDIR" ]; } 2>/dev/null || { echo "cannot write $OUTDIR - using /tmp"; OUTDIR=/tmp; }
OUT="$OUTDIR/profile_$(date +%Y%m%d-%H%M%S).txt"
NAME="${Q36_ENGINE_NAME:-qwen36}"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT

echo "waiting for the engine process ($NAME) ..."
PID=""; for _ in $(seq 1 600); do PID="$(pgrep -nx "$NAME" || true)"; [ -n "$PID" ] && break; sleep 1; done
[ -n "$PID" ] || { echo "no $NAME process found after 10 min"; exit 1; }
echo "engine pid $PID"

cpu_ticks() { awk '{print $14+$15}' "/proc/$1/stat" 2>/dev/null; }
if [ "$NOW" -eq 0 ]; then
  # "busy" = 75% of min(cores, the engine's OMP threads): 300% on a 4-core box running 4 threads
  NCPU="$(nproc)"; THR="$({ tr '\0' '\n' < "/proc/$PID/environ" | sed -n 's/^OMP_NUM_THREADS=//p'; } 2>/dev/null)"
  case "$THR" in ''|*[!0-9]*) THR="$NCPU" ;; esac
  [ "$THR" -lt "$NCPU" ] && NCPU="$THR"
  WANT=$(( NCPU * 75 ))
  echo "waiting for the steady compute-bound phase (>=${WANT}% CPU for 8 s in a row) ..."
  hits=0; HZ="$(getconf CLK_TCK)"
  while kill -0 "$PID" 2>/dev/null; do
    a="$(cpu_ticks "$PID")"; sleep 2; b="$(cpu_ticks "$PID")"
    pct=$(( (b - a) * 100 / (2 * HZ) )) 2>/dev/null || pct=0
    if [ "$pct" -ge "$WANT" ]; then hits=$((hits + 1)); else hits=0; fi
    [ "$hits" -ge 4 ] && break
  done
  kill -0 "$PID" 2>/dev/null || { echo "the engine exited before reaching a steady phase"; exit 1; }
fi

{
  echo "q36 profile — $(date)   engine pid $PID   $SECS s"
  echo "engine: $(readlink -f "/proc/$PID/exe")  md5 $(md5sum "/proc/$PID/exe" 2>/dev/null | cut -d' ' -f1)"
  echo "threads: $(ls "/proc/$PID/task" | wc -l)   cpu MHz now: $(awk '/cpu MHz/ {printf "%d ", $4}' /proc/cpuinfo)"
  echo "engine env: $({ tr '\0' ' ' < "/proc/$PID/environ"; } 2>/dev/null | grep -oE '(Q36_[A-Z0-9_]+|HOT_FRAC|GRIFFIN|TRUNK_[A-Z]+|OMP_NUM_THREADS)=[^ ]*' | tr '\n' ' ')"
} > "$OUT"

METHOD=""
if command -v perf >/dev/null 2>&1 && perf record -F 49 -p "$PID" -o "$W/probe.data" -- sleep 0.3 >/dev/null 2>&1; then METHOD=perf; fi
[ -z "$METHOD" ] && command -v gdb >/dev/null 2>&1 && METHOD=gdb
if [ -z "$METHOD" ]; then
  echo "need perf or gdb:  apt-get install -y gdb      (or: apt-get install -y linux-tools-common linux-tools-\$(uname -r))"
  exit 1
fi
echo "method: $METHOD" | tee -a "$OUT"
echo "sampling $SECS s ..."

if [ "$METHOD" = perf ]; then
  perf record -F 199 -g -p "$PID" -o "$W/perf.data" -- sleep "$SECS" >/dev/null 2>&1
  { echo; echo "== functions by self time (perf) =="; perf report -i "$W/perf.data" --stdio --no-children --percent-limit 0.4 -g none --sort symbol 2>/dev/null | grep -v '^#' | grep -v '^$' | head -40
    echo; echo "== by library =="; perf report -i "$W/perf.data" --stdio --no-children --percent-limit 1 -g none --sort dso 2>/dev/null | grep -v '^#' | grep -v '^$' | head -10; } >> "$OUT"
else
  : > "$W/leaf"; : > "$W/incl"; n=0; END=$(( $(date +%s) + SECS ))
  while [ "$(date +%s)" -lt "$END" ] && kill -0 "$PID" 2>/dev/null; do
    gdb -p "$PID" -batch -ex "set pagination off" -ex "thread apply all bt 8" 2>/dev/null > "$W/s.txt"
    # one block per thread; skip threads parked in a wait (they are not working)
    awk -v LEAF="$W/leaf" -v INCL="$W/incl" '
      function fn(l,   s) { s=l; sub(/^#[0-9]+ +/,"",s); sub(/^0x[0-9a-f]+ in /,"",s); sub(/ \(.*/,"",s); sub(/ from .*/,"",s); return s }
      /^Thread / { if (nf>0) flush(); nf=0; delete seen; wait=0; next }
      /^#[0-9]+ / { f[nf++]=fn($0); if ($0 ~ /futex|pthread_cond|clock_nanosleep|__poll|epoll_wait|sem_wait|do_wait|gomp_barrier_wait|gomp_team_barrier_wait|gomp_thread_start/ && nf<=2) wait=1 }
      function flush(   i) { if (wait) { print "(idle/waiting)" >> LEAF; return }
        print f[0] >> LEAF; for (i=0;i<nf;i++) if (!(f[i] in seen)) { seen[f[i]]=1; print f[i] >> INCL } }
      END { if (nf>0) flush() }' "$W/s.txt"
    n=$((n + 1))
  done
  total=$(wc -l < "$W/leaf")
  { echo; echo "== $n gdb snapshots, $total thread samples =="
    echo; echo "== where threads were executing (innermost function) =="; sort "$W/leaf" | uniq -c | sort -rn | head -25 | awk -v t="$total" '{printf "%6.1f%%  %s\n", 100*$1/t, substr($0, index($0,$2))}'
    echo; echo "== inclusive (function anywhere in the stack; the work a function is responsible for) =="; sort "$W/incl" | uniq -c | sort -rn | head -25 | awk -v t="$total" '{printf "%6.1f%%  %s\n", 100*$1/t, substr($0, index($0,$2))}'; } >> "$OUT"
fi
echo; cat "$OUT"; echo; echo "saved: $OUT"
