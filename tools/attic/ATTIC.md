# tools/attic/ — superseded patch scripts (E1 drift audit, 2026-10-19)

The scripts here were the historical per-feature engine patchers. The E1 audit
(`docs/ROADMAP.md` E1, now CLOSED) found that they no longer reproduced the
committed engine snapshots in any application order:

- `patch_qwen_kv8_p2.py` — KV8 "part 2"; part 1 was never committed, so this
  applied to stock but produced a non-historical tree.
- `patch_flash_qwen36.py` — flash **v2** patcher; the shipped engine carries
  flash **v1** (the `premap` snapshot), so its anchors no longer matched.
  Kept here as the design for a future flash rebuild.
- `patch_hotneuron.py`, `patch_hotcal_v4.py`, `patch_griffin.py`,
  `patch_mapfast.py`, `patch_pagebatch.py` — drifted after the `pretq`
  snapshot; their feature edits are interleaved inside single diff hunks, so
  the historical per-feature split is not recoverable from the snapshots.

Their combined effect is reproduced **byte-exactly** by the regenerated
patchers in `tools/`:

    patch_kv8_flash_v1.py   orig   -> premap   (KV8 both parts + flash v1)
    patch_hotstack_v1.py    prehot -> pretq    (hot-neuron + cal + GRIFFIN + mapfast/pagebatch)

Full chain (verified by `tools/verify_engine.sh` and CI on every push):

    orig -> kv8_flash_v1 -> premap -> trunkmap -> trunkmap_v2 -> prehot
         -> hotstack_v1 -> pretq -> kvtq -> presm -> softmax -> qwen36.c

Do not delete these files: they document intent (especially flash v2) and the
pre-regeneration history. Do not apply them: they will not reproduce the
snapshots.
