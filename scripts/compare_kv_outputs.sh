#!/bin/bash
# compare serve-mode outputs f32 vs KV8 vs KV_TQ4 (lossiness check)
cd ~/colibri/c
run_one() {
  local kv="$1"
  SNAP=${Q36_HOME:-/home/dylan}/synth_models/synth_std CTX=2048 KVSAVE=0 USAGE_SAVE=0 CHAT_TEMPLATE=0 \
  NGEN=48 COLI_TEMP=0 SERVE=1 CAP=8 CAP_RAISE=0 DIRECT=1 PIPE=1 $kv \
  ./colibri 8 16 16 2>/dev/null | tr "\n" " " | md5sum | cut -d" " -f1
}
A=$(run_one "")
B=$(run_one "KV8=1")
C=$(run_one "KV_TQ=4")
echo "f32   md5: $A"
echo "kv8   md5: $B"
echo "kvtq4 md5: $C"
[ "$A" == "$B" ] && echo "kv8 IDENTICAL to f32" || echo "kv8 DIFFERS from f32 (lossy, expected)"
[ "$A" == "$C" ] && echo "kvtq4 IDENTICAL to f32" || echo "kvtq4 DIFFERS from f32 (lossy, expected)"
