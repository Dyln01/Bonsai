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
- [x] **Phase 2 prep:** MTP design contract pinned (`docs/MTP_DESIGN_QWEN36.md`, `633d35a`) + synthetic head generator (`scripts/gen_mtp_head_synth.py`, `bcedf90`, shape fix `6d690e0`)

## Open — phases

- [ ] **Phase 2b:** MTP head engine integration — `tools/patch_mtp.py` per the pinned plan in
      `docs/MTP_DESIGN_QWEN36.md` §"NEXT": meta keys, `MTPHead` struct, loader, head forward,
      hidden capture at the `step_ex` seam; knobs `Q36_MTP=1`, `Q36_MTP_DRAFTS=1|2|3|0`,
      `Q36_MTP_FORCE=accept|mixed|cycle|rowN`; gates: forced-accept oracle byte-exact,
      knob-unset stock byte-identity, draft-cost battery; then COMMANDS §10 + README update.
- [ ] **Phase 3:** absorbed-score attention (FlashMLA-style, fold q·K per GQA group) — fixes
      KV8/TQ decode cost (KV8 −43%, TQ4 −89%, polar −97% in the KVTQ matrix) + O(T) long-ctx
      decay. Design-contract doc first (house practice), then implementation, then rerun the
      KVTQ 8-arm matrix + flash battery.
- [ ] **Phase 4:** windowing (sliding active-neuron cache, Apple LLM-in-Flash analog) +
      dynamic predictor (Deja Vu/ShadowLLM class, must beat GRIFFIN or stay a documented
      negative) + cross-engine KV8/TQ ports (GLM/qwen38) with per-engine parity gates.
      Grounding: `docs/RESEARCH_MAP_DENSE.md`.
- [ ] **Phase 5:** real-weights validation — GATED on user sign-off (breaks the no-download
      rule). Protocol doc first (eval set + checksums, perplexity arms, MTP acceptance rate,
      KV-tier quality neutrality, download budget + license check); execution only after
      sign-off; refresh README projected rows afterwards.

## Open — engineering / hygiene

- [ ] **E1 — patch-script regeneration audit.** `tools/verify_engine.sh` replays the segments
      that reproduce (premap→prehot, pretq→presm, presm→final) and md5-pins everything else.
      Two segments do NOT replay with committed patchers: orig→premap (the KV8 **part-1**
      patcher was never committed and `patch_flash_qwen36.py` is now v2 while `premap`
      carries flash v1) and prehot→pretq (hotneuron/griffin/mapfast/pagebatch/hotcal
      patchers drifted after the pretq snapshot). Close by re-deriving each patcher's
      anchors against the committed snapshot chain so orig→final replays end-to-end;
      until then the engine/ snapshots + `tools/patch_manifest.json` are the source of
      truth. (Found 2026-10-19; README reproducibility claim corrected in the same commit.)
- [ ] **E2 — vendor the missing harness files.** `docs/REPORT.md`'s I/O-fidelity shim
      (`harness/iothrottle.c`) and `scripts/gen_qwen27_synth.sh`'s generator module
      (`make_synthetic_model.py`) are referenced but not in this repo. Commit them (or
      link public copies) so the harness is reproducible; errata headers added in the
      meantime.
- [ ] **E3 — milestone tags.** Tag `phase-1a`, `phase-1b`, `phase-2a`, `softmax` so engine
      backups can be pruned and regressions bisected without archaeology.
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
