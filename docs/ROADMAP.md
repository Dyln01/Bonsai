# Bonsai roadmap — the phase board, in-repo

**Living document.** This is the authoritative checklist (previously it existed only in
session notes). Update it in the same commit that closes/opens an item.
Format: GitHub task lists, so `[x]` renders as a checkbox.

## Shipped

- [x] **Phase 0:** Bonsai rebrand (remote URL, README, rebase on user's GitHub edits) — `70bc253`
- [x] **Phase 1a:** KV_TQ survey (kv_tq.h codecs, qwen36 KV8 seams, GLM wiring, container shapes)
- [x] **Phase 1a:** KV_TQ port — `patch_kvtq.py` applied (20 edits), engine builds, parity gates all green — `15bb6e7`
- [x] **Phase 1a:** launcher `Q36_KVQ` knob (f32|kv8|tq4|tq3) + engine synced to repo (md5-verified)
- [x] **Phase 1a:** 8-arm battery on synth_qwen27 — all rc=0; RSS ladder 805.6/673.3/649.9/641.3 MB, decode 46.9/26.6/5.3/1.25 tok/s (honest timers) — `d70b43d`
- [x] **Phase 1a:** ship — docs/KVTQ_QWEN36.md + results JSON + README updated; `15bb6e7`/`98073a5`/`d70b43d`
- [x] **Phase 1b:** kv_prefix reuse — MEASURED + shipped: 93% reuse through mux slot swaps, turns 2+ 10-12x faster; agent contract documented; `Q36_SLOTS` launcher knob — `67976ca`
- [x] **User ask:** full install guide — docs/INSTALL.md (Linux + Windows/WSL2), verified by executing its parity commands — `98073a5`
- [x] **User ask:** full command documentation — docs/COMMANDS.md: launcher, ~30 engine knobs, wire protocol, generators, drivers, patch order, parity gates — `b219e21`
- [x] **Softmax optimization:** `Q36_FAST_SOFTMAX=1` — 9.5-11.5x kernel, attn phase −6.5% (5/5 pairs), TTFT neutral, more accurate than stock; flash-site negative result isolated + documented — `3aa65fa`
- [x] **Phase 2a:** stock-safety regression — stock run rc=0 (RSS 3.03 GB) with the MTP head file present; MTP-unaware engines ignore `mtp.*` + meta keys, by construction and by measurement — `94ab7c1`
- [x] **Calibrate UX fix (2026-10-19, `test` branch):** `calibrate` looked hung on the mini-PC
      ("stops after warm log, >1 h, never quits"). Root cause: full-compute (`HOT_FRAC=1.0`)
      prefill over the WHOLE corpus at ~1-2 s/token on dense 27B × 4 cores, engine stderr
      swallowed to `/tmp/calib.err`, and the profile written ONLY at clean exit (a `^C` lost
      everything). Launcher now: streams stderr live, caps the corpus at `Q36_CAL_TOKENS`
      (default 1024; `0` = whole file), prints an ETA banner from the container geometry,
      and fails loudly with a pointer if no profile lands. Engine-side periodic save +
      progress lines tracked as **E7**.
- [x] **Phase 2 prep:** MTP design contract pinned (`docs/MTP_DESIGN_QWEN36.md`, `633d35a`) + synthetic head generator (`scripts/gen_mtp_head_synth.py`, `bcedf90`, shape fix `6d690e0`)

## Open — phases

- [x] **Phase 2b-1 — MTP head loader + knobs (2026-10-19, `test` branch).**
      `tools/patch_mtp.py` (7 anchored edits, backup `.premtp`): meta keys
      `mtp`/`mtp_layers`, knobs `Q36_MTP=1` / `Q36_MTP_DRAFTS=0..3` /
      `Q36_MTP_FORCE`, dense-head f32 loader via `st_find` on `m->S`
      (model-mtp.safetensors is already indexed — `st_scan_dir` takes every
      `*.safetensors`), refuse-gates (headless, meta-without-tensors, MoE head),
      report line `[qwen36 MTP] head loaded`, `q36_mtp_free` in the destroy path.
      CI gates: refusal rc!=0 with message; knob-on loads + token stream
      byte-identical to stock (head inert in part 1).
- [x] **Phase 2b-2 — MTP drafting + verify, dense head (2026-10-19, `test`).**
      `tools/patch_mtp_draft.py` (11 anchored edits, backup `.predraft`): head forward in the
      S=1 dialect (own f32 K/V rows, qk-norm, `rope_head_partial`, GQA scores over the head's
      standing rows, `softmax_row`, attn output gate, o_proj, y-ring); hidden capture at
      `step_ex`'s lm_head seam (ring of 8 settled rows); `q36_mtp_propose` (d1 from
      embed(tok)+hidden[pos-1], deeper drafts from the head's own hidden); source choice
      beside lookup via `spec_gate_pick` values; `sp->cur_src` in settle + report; head-row
      rollback = kv_len trim; `[qwen36 MTP] run:` report line. CI gate 3: greedy stdout
      byte-identical to stock for `Q36_MTP_DRAFTS` 0..3 with `COLI_LOOKUP=0 COLI_SPEC_GATE=0`
      (random-weight head => every verify rejects => rollback exercised every verify).
- [x] **Phase 2b-3a — force-oracle gates (2026-10-19, `test`).** `patch_mtp_oracle.py`:
      `Q36_MTP_FORCE=accept|mixed|cycle|rowN` drives the MTP proposal slot from ref.json's
      oracle ids; `DUMP_IDS` bootstraps ref.json in CI. Gates green: forced-accept ⇒
      acceptance 100.0% (7/7) AND the id sequence equals stock (the design's "reproduces
      the oracle text exactly" — the correctness proof of verify + head-row rollback);
      mixed/cycle ⇒ 60%/45.5% acceptance with the sequence STILL exact under rejections.
      Documented nuance: final logits may differ from a plain step when the last pick is
      answered from a verify batch row (pre-existing S>1 fp-order behavior, shared with
      accepted lookup drafts) — hence sequence identity, not logit bits, is the invariant.
- [x] **Phase 2b-3b — MoE head + head-MLP fidelity fix (2026-10-19, `test`).**
      `patch_mtp_moe.py`: router softmax -> group-limited top-k (mirrors `moe_ex`) ->
      renormalised mix of unpacked-int4-gs64 experts + optional sigmoid-gated shared
      expert; ALSO fixes a part-2 fidelity gap found in review: the head row ended at
      o_proj — the post-attention norm + MLP stage now runs per contract (dense and MoE).
      CI: small MoE synth (`SYNTH_*` geometry overrides on gen_coder30_synth.py, 8L x 32E,
      same code paths) — head loads dense=0, drafting-on sequence == stock, forced-accept
      sequence == stock at 100% acceptance. NOT covered in CI: shared-expert head path
      (SI>0; no synthetic MoE container carries shared experts yet) and resident-int4
      experts (3b unpacks to f32 at attach — fine for synth heads, too heavy for real
      35B-class heads on 8 GB).
- [ ] **Phase 2b-4 — draft-cost battery on real hardware + report.** Runner ready:
      `tools/run_battery.sh <model> <prompt>` (arms A f32 / B kv8 / C tq4-absorb0 /
      D tq4-absorb1 / E kv8+MTP drafts / F kv8+MTP forced-accept; emits
      results/battery_<host>_<ts>.json + summary table). Per-draft cost `d`
      (the gate's input), verify T(n) by row count, tokens/forward under forced acceptance,
      on the mini-PC; -> `results/results_mtp_q36.json` + `docs/MTP_QWEN36.md`.
- [x] **Phase 2b-5 — resident-int4 head experts + SI>0 gate (2026-10-19, `test`).**
      `patch_mtp_i4.py`: merged int4-gs packed bytes + scales stay resident as stored
      (dequant-on-read `mtp_gemv_i4gs`, OpenMP-safe, deterministic) — attach RAM no longer
      scales with experts x f32, so real 35B-class heads fit the 8 GB envelope; dead
      expansion helper dropped. `gen_coder30_synth.py` gains `SYNTH_SI` shared experts;
      CI gate: SI>0 MoE head loads (dense=0) and drafting-on sequence == stock.
- [x] **Phase 3a — GQA-absorbed decode tiles (2026-10-19, `main`).** `patch_gqa_absorb.py`:
      dequant each K/V tile ONCE per (kv-head, row) and serve all q_per_kv heads of the
      group — the per-head loop paid the same dequant q_per_kv times (2 on synth_qwen27,
      5 on the real 27B). Lossless by construction; proven bit-identical by A/B binary
      DUMP_IDS gates (f32/kv8/tq4 @16k + kv8 @68.4k fixture, all rc=0, mtp-kv8 coexists).
      Measured attention phase: KV8 −34.5%, TQ4c1 −47.0% @16k; KV8 −27.4% @68.4k; f32
      control unchanged; RAM unchanged. `docs/GQA_ABSORB_QWEN36.md` +
      `results/results_gqa_absorb_q36.json`.
- [x] **Phase 3b-rot — absorbed-rotation TQ decode (2026-10-19, `absorb`).**
      `patch_tq_absorb.py` per docs/ABSORB_ROT_QWEN36.md: codec-1's fixed rotation moves out
      of the per-row loop (rotate queries once per head-step, dot raw rotated-space levels,
      accumulate V in rotated space, unrotate once) — per-row decode cost becomes KV8-class
      at half KV8's bytes. `Q36_TQ_ABSORB=0` restores 3a verbatim; CI parity gate: cosine
      >= 0.99999 + identical id sequences, absorb off vs on.
- [x] **Phase 3b-prefill — absorbed rotation in the flash tile loop (2026-10-19).**
      `patch_tq_absorb_prefill.py` (same `Q36_TQ_ABSORB` knob): queries rotate once per
      (s0,kvh) block, K/V dequant to rotated-space levels per block, accumulators stay
      rotated through the online softmax, one unrotate per (row,head). Covered by the
      absorb 0/1 id-identity CI gate (flash on by default).
- [x] **Phase 3b-scan design contract (2026-10-20):** `docs/SCAN_DECAY_QWEN36.md` —
      two-tier deterministic decode scan (local window + measured sink blocks +
      salience-gated 128-row blocks, ε-budget), knobs default `full`, correctness
      interactions pinned (spec snapshots, prefix reuse, KV-tier composition), four
      pre-registered gates incl. the ≤+0.5% ppl band and the ≥2x 68.4k speed gate.
      Implementation open; OasisKV draft-prefetch noted as a post-2b-4 composition.
- [ ] **Phase 3b remainder:** implementation of the scan contract above, plus the
      measured KVTQ battery arms with Q36_TQ_ABSORB=0/1 on real hardware (fold into the
      2b-4 battery run).
- [ ] **Phase 4:** windowing (sliding active-neuron cache, Apple LLM-in-Flash analog) +
      dynamic predictor (Deja Vu/ShadowLLM class, must beat GRIFFIN or stay a documented
      negative) + cross-engine KV8/TQ ports (GLM/qwen38) with per-engine parity gates.
      Grounding: `docs/RESEARCH_MAP_DENSE.md`.
- [ ] **Phase 5:** real-weights validation — GATED on user sign-off. The contract is
      written and reviewable: `docs/VALIDATION_PROTOCOL_QWEN36.md` (checkpoints + license
      checks, wikitext-2 ppl arms A-I with pre-registered acceptance bands, mini-PC
      mechanics M1-M5, resumable session procedure, sign-off checkbox). Execution begins
      only after the checkbox is ticked; `scripts/make_wikitest_ref.py` lands with the
      session commit.
- [x] **E1 — patch-script regeneration audit (CLOSED 2026-10-19).** Finding: the KV8
      part-1 patcher was never committed, the flash patcher had drifted to v2 while the
      engine carries v1, and the hotneuron/hotcal/griffin/mapfast/pagebatch patchers no
      longer replayed in any order (feature edits interleaved inside single diff hunks).
      Resolution: two byte-faithful regenerated patchers (`patch_kv8_flash_v1.py`,
      `patch_hotstack_v1.py`) derived from the committed snapshots; the FULL chain
      orig→premap→prehot→pretq→presm→final now replays byte-exact under
      `tools/verify_engine.sh` and in CI. Superseded per-feature scripts kept in
      `tools/attic/` with `ATTIC.md`. The README reproducibility claim is true again.
- [x] **E2 — harness files vendored/reconstructed (2026-10-19).** `scripts/make_synthetic_model.py`
      reconstructed from its contract of use in `gen_qwen27_synth.sh` (with the legacy
      `data_offsets` bug — see `fix_qwen27_container.py` — fixed; CI self-check).
      `scripts/iothrottle.c` reconstructed from the `REPORT.md` §0 spec (CI: build +
      bandwidth smoke). Originals were never committed; reconstructed files are labeled as such.
- [x] **E3 — milestone tags.** `phase-1a`, `phase-1b`, `phase-2a`, `softmax` annotated tags
      pushed; engine backups prunable once E1 closes.
- [x] **E7 — engine-side progress + incremental profile save (2026-10-19, `bar`).**
      `patch_calib_bar.py`: `[prefill] p=.. layer=i/L t=.. eta=..` stderr line per layer of
      every S>1 forward (knob `Q36_PROGRESS=0`), periodic `q36_hot_profile_save()` every 8
      layers when `HOT_PROFILE` is set (a killed calibrate keeps a partial profile), and the
      launcher renders it: inline bar on `calibrate`/`bench` + `q36-minipc watch` for a
      second terminal. Stderr-only => CI identity gates prove compute bit-neutrality.
- [x] **E8 — TRUNK_MAP int4 scale clobber FIXED (2026-10-19).** `load_tq`'s mmap branch
      loaded per-row `.qs` scales then nulled them; `matmul_map_i4` ran with rs=1.0 on every
      dense int4 TRUNK_MAP run since TRUNK_MAP shipped. Fixed by `patch_mapsc.py` (chain hop
      `.premapsc`); CI gate: a scales-=1.0 container copy must now differ from the real one
      (pre-fix they matched exactly). Errata: pre-fix dense measurements (LOWRAM/BENCH/student
      ratios) remain valid as RELATIVE A/B within one container; absolute quality and the
      hot-neuron energy ranking were computed on scale-less weights — re-measure in the 2b-4
      battery. Serve-mux on packed containers also needs TRUNK_MAP=1 (launcher always set it;
      the prefix driver now does too).
- [ ] **E4 — launcher `doctor` upgrades.** Report `Q36_FAST_SOFTMAX` engagement (AVX2 build?),
      `Q36_KVQ` tier, `Q36_SLOTS`, and `meta.mtp` presence in the container.
- [ ] **E5 — prefix-reuse regression gate.** Promote `scripts/exp_prefix_q36.py` to a
      fixed-seed gate with expected reuse % once CI can run the engine.
- [ ] **E6 — flash-site fast-exp retry.** The online-softmax negative result gets one retry
      with the block-rescaled form before it is considered closed (`engine/softmaxbench_q36.c`).

## Rules that stay

- No real checkpoint downloaded (Phase 5 excepted, by explicit sign-off).
- Every number backed by a committed `results/*.json`; decode numbers only from honest
  `[timers]` lines.
- Design-contract doc before implementation; negative results documented, not buried.
- Parity gates must pass in CI (`.github/workflows/gates.yml`) on every push.
- Synthetic containers are MECHANICAL oracles (crash-freedom, state-machine and
  rollback exactness, sequence identity across engine variants), NOT numeric ones:
  random-weight deep nets explode or degenerate by construction (proved 2026-10-19:
  degenerate-head student made logit-magnitude gates vacuous; real-head student
  explodes to 1e8 residuals). Numeric truth lives in the C parity gates
  (kv8_parity, kv_tq_parity, softmaxbench, tq_absorb_parity) and in measured
  batteries on real hardware. The `STUDENT_REALHEAD` generator flag stays as an
  experiment, unused by CI.
