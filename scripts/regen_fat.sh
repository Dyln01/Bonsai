#!/bin/bash
# Right-sized fat-trunk regen: the point of this model is an oversized DENSE
# TRUNK relative to RAM_GB, not many experts. 12 layers x hidden 4096 gives a
# ~1.2 GB int4 trunk; experts 128 x moe_inter 512 keeps the expert set ~3.3 GB.
set -e
rm -rf ~/synth_models/synth_fat_trunk
cd ~/colibri-research/synthgen
python3 make_synthetic_model.py --out /home/dylan/synth_models/synth_fat_trunk --seed 7 \
  --layers 12 --experts 128 --topk 8 --hidden 4096 --moe-inter 512 --heads 32 \
  --dense-inter 2048 --kv-lora 256 --q-lora 256 --vocab 4096 --first-dense 1 --shared 1 \
  2>&1 | tail -4
echo "=== fat trunk size ==="
du -sh ~/synth_models/synth_fat_trunk

