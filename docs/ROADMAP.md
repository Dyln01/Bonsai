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
- [x] **Phase 2b-4 — draft-cost battery + report (2026-10-20, `main`).** Six arms,
      all rc=0 (`tools/run_battery.sh` -> `results/results_mtp_q36.json` +
      `docs/MTP_QWEN36.md`). Gate calibration: T(1)=16.3 ms, T(2)=1.73x, T(3)=2.64x
      (sublinear row batching); 123 drafts / 62 verifies at drafts=2; forced-accept
      settles 3 tok/forward at +11% worst-case per-token overhead; natural
      acceptance 0.0% on random weights (expected — meaningless by design; real
      acceptance is a pre-registered Phase 5 arm). Same run lands the 3b-rot
      end-to-end numbers: TQ4 decode 53.39 -> 24.49 ms/tok (-54.1%), 3.05x -> 1.40x
      KV8, RAM win intact. Operating point: 4-core WSL2 (the standing battery
      point); the runner reproduces the whole battery on the literal mini-PC with
      one command.
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
- [x] **Phase 3b-scan — two-tier deterministic decode scan (2026-10-20, `main`).**
      `engine/scan_decay.h` + `patch_scan_decay.py` (chain 16 hops, backup .prescan):
      window + measured sinks + salience-gated 128-row blocks; table uint64
      fixed-point so schedule(dynamic) cannot break bit-reproducibility; seeding
      = settle row + prefill chunks (no spec snapshots needed). Gates on synthetic:
      scan-off id-identity PASS (kv8+tq4), two-run determinism PASS, machinery
      ceiling 3.2% rows -> 11.9x attention at 16k (EPS=1.0), EPS=0.05 engages at
      54% rows / -14% attention; gate 2 (ngram self-check) = documented NEGATIVE on synthetic, RECLASSIFIED 2026-10-22 as expected physics (no induction heads on random weights; decode salience reinforces the selected set by design) — moved to the Phase 5 arm bundle
      (random weights = pure causal-coverage gradient, zero content signal;
      needs real weights or a structured container); quality + real-speed gates
      pre-registered as Phase 5 arms. Knob Q36_SCAN default full.
- [x] **Phase 4a — predictor + window: documented NEGATIVE on synthetic
      (2026-10-20, `main`).** `Q36_PROFILE_TOK` per-token dumps
      (`patch_profile_tok.py`, chain 17 hops) + `scripts/exp_predictor_q36.py`
      (three-way offline gate). Measured on student_1b_i4cal, disjoint
      4096-token corpora: static-A == static-B == GRIFFIN == **1.0000**
      energy coverage — the per-token top-K set is CONSTANT (norm-driven,
      content-independent on random weights); the Deja Vu premise is absent
      by construction, the ridge fit learns the constant set (0.9996), and
      the window has no per-token sets to cache. Runtime work NOT unlocked
      per the pre-registered rule; the machinery (dump + fit + gate) is
      retained as the ready-to-run Phase 5 arm on real containers.
      `docs/PREDICTOR_QWEN36.md` + `results/results_predictor_q36.json`.
- [x] **Phase 4b design contract (2026-10-20):** `docs/WINDOWING_QWEN36.md` —
      per-layer resident neuron-row window over the dense gather path (FIFO,
      byte-identical copies so output is cache-history-independent), value cases
      pinned honestly around 4a's negative (real weights + GRIFFIN-on-HDD), four
      pre-registered gates (offline sim on 4a dumps → CI id-identity → /proc/io
      read-bytes cut on the mini-PC → Phase 5 ppl neutrality), three implementation
      slices scoped.
- [x] **Phase 4b slice 1 (2026-10-22, `win`):** `patch_window.py` window cache +
      **dense GRIFFIN observe fix** found while gating it: `q36_griff_observe` ran only in
      the MoE expert path, so on dense containers GRIFFIN never built a set, `gs` stayed
      NULL, and the gather path (hence the window) was unreachable — the README's
      "GRIFFIN adapts per-prompt on unmodified containers" was silently a static-prefix
      fallback on dense. Observe now runs on the dense gate activations (prefill rows).
- [x] **Phase 4a RE-RUN (2026-10-22):** done with live GRIFFIN — verdict FAIL again
      (+0.00 vs both baselines) but diagnosis revised: coverage SATURATES at 1.0 for every
      selector on synthetic (norm-dominant energy mass), while sets DO vary (jacc 0.17).
      4a's original constant-set claim was twice an artifact (inert GRIFFIN, then metric
      saturation). G2 discriminates only on real weights — Phase 5 arm unchanged.
      `results/results_predictor_rerun_q36.json` + PREDICTOR doc addendum.
- [x] **Phase 4b slice 2 (sim):** `scripts/exp_window_sim.py` + `patch_tokdump_w.py`
      (dumps went silent once gather went live — the writer only sat in the per-row
      fallback; narrowed-width writer keeps record format). Synthetic curve in
      `results/results_window_sim_q36.json`; CI replays it per push. Side finding:
      per-token sets DO vary on synthetic once GRIFFIN is alive (jacc 0.17) — 4a's
      constant-set negative was an inert-GRIFFIN artifact; predictor re-run queued.
- [x] **Phase 4b slice 3 (field verdict 2026-10-25, E14):** battery arms
      W0/W256/W1G (GRIFFIN × window) + per-arm `File system inputs` read-byte
      accounting shipped in `run_battery.sh`; measured cold-cache (BATTERY_WARM=0)
      on the HDD mini-PC in the 2026-10-25 overnight matrix
      (`results/battery_llm-030_20261025-133007.json`). **Gate 3: FAIL —
      documented negative.** W256/W1G cut fs_read 0.047%/0.018% vs W0
      (71.791/71.812 vs 71.825 GB) against the pre-registered ≥50% bar. The
      reason is mechanical, not statistical: the window CANNOT ALLOCATE on
      q27_i4 — `q36_win_ensure` requires int4-packed MLP maps
      (`map_len == D*I/2`) and the dense trunk is F16 (E13); `hits 0 misses 0`
      on both window arms is that signature (lookup returns before counting
      when `win == NULL`). W0≡W256≡W1G were one configuration thrice. And at
      the field-forced n_new=2 (ONE decode step) the ≥50% bar was
      arithmetically unreachable anyway: decode re-read is ≤33% of total
      bytes and steady-state hits need ≥2 steps. `Q36_WIN_MB` stays
      default-off (engine default 0; the launcher's rotational auto-256 never
      shipped and now must not). Re-gate conditions in `docs/WINDOWING_QWEN36.md`.
- [x] **Phase 4c — cross-engine KV8/TQ ports CLOSED (2026-10-22).** qwen38: KV8 +
      KV_TQ/absorb ports, gates green on a CI-sized qwen4_exp_text container (slices
      1-3). glm53: dialect appendix B, synthetic container the stock engine runs first
      try, latent-TQ port (`patch_kvq_glm53.py`) with TF-id gates: tq-off == stock,
      tq-on self-deterministic, absorb 0v1 78/78 (slice 4). Three engines now share the
      tier ladder; per-engine patchers + own backups; qwen36 chain untouched throughout.
- [x] **Phase 4 CLOSED** — 4a predictor (negative, diagnosis corrected), 4b windowing
      (cache + sim merged; slice-3 field verdict landed 2026-10-25: read-matrix flat,
      window unallocatable on the F16 trunk, knob stays default-off — E14), 4c ports
      (qwen38 + glm53). Tag `phase-4`.
- [~] **Phase 5 — IN PROGRESS (sign-off recorded 2026-10-20, W1 first).**
      Session kit shipped: `tools/run_phase5.sh` (resumable per-arm ppl runs A-I on
      the engine's PPL=1 path, M1 determinism, M3 calibrate+sort optional, M4 RSS,
      M5 via run_battery.sh when meta.mtp; manifest sha256s; summary JSON with the
      §3 bands applied). First field attempt 2026-10-26 (`cfc4e91`): three
      manifest-only tombstone sessions — the ref builder ran the engine in the
      stock dialect (E15) and the scripted per-token tf-nll would have cost
      ~52 days/arm (E16). Kit fixed same day: streaming-dialect ref build +
      WIKI fail-fast + cached manifest/M1 + `--len` math (E15), and
      `Q36_TF_CHUNK` chunked scoring with a CI parity gate + `PHASE5_LEN`
      default 64 (E16) — session budget now ~28–32 h. Re-armed: resume into
      `results/realweights_20261026-1014` (manifest cached, no
      `PHASE5_FORCE` needed — no arm ever ran). Report doc
      `docs/REALWEIGHTS_QWEN36.md` gets written from `summary.json`.
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
- [x] **E4 — launcher `doctor` upgrades (2026-10-19, `b5ca30d`).** doctor reports
      `Q36_FAST_SOFTMAX` engagement, `Q36_KVQ` tier, `Q36_SLOTS`, `meta.mtp` presence,
      plus frac/griffin/cal state.
- [x] **E5 — prefix-reuse regression gate (2026-10-19, `b5ca30d`).** `exp_prefix_q36.py`
      parameterized (PREFIX_*) and gated in engine-build CI: mux_on reuses ≥80% at
      turns≥2, mux_off never.
- [x] **E6 — flash-site fast-exp retry (2026-10-19, `d948558`; measured negative).**
      The block-rescaled retry ran as a gated microbench (`[online-softmax E6]` line,
      f64 error gate hard): fast-exp+exact-rescale = **0.86x** of stock expf on the
      online 16k loop (rescale/memory-bound, as predicted) — not adopted; the
      flash-site negative result is closed with numbers.

- [x] **E9 — HDD/small-RAM streaming mode (2026-10-20).** Field report: calibrate
      "frozen at layer 48 for 2 h" on an 8 GB mini-PC with a **rotating disk**. Cause:
      TRUNK_MAP's greedy i+2 WILLNEED prefetch + unbounded page cache on a box that can
      neither hold the trunk nor seek three interleaved streams — reclaim/seek thrash,
      not a hang. `patch_trunk_stream.py` (`Q36_TRUNK_STREAM`, auto-on with HOT_PROFILE):
      no greedy prefetch + MADV_DONTNEED each layer's ten slabs after its MoE, bounding
      the cache footprint to ~one layer. Launcher also probes
      /sys/block/*/queue/rotational for the ETA bandwidth default (150 vs 1500 MB/s).
      CI: stream-mode id-identity gate. NVMe boxes unchanged (knob 0, default off
      outside calibrate).

- [x] **E10 — hot_sort v2 reader bug, found by its first production run (CLOSED
      2026-10-23).** Field report: `q36-minipc sort ~/Models/q27_i4` died with
      `cannot reshape array of size 178263040 into shape (17408,5120)`. Cause
      (proven by miniature repro — v2 on the fixture reproduces the exact
      signature): v2's `read_st` unpacked safetensors `data_offsets` as
      (offset, LENGTH); the format stores (begin, END). Every tensor not at data
      offset 0 was over-read by its own `begin`: the real shard's F16 gate_proj
      sits at begin=178,268,160 B (one (I,H) F16 matrix + one (H,) F16 vector
      ahead of it), so v2 got (begin+span)/2 = (2·17408+1)·5120 = 178,263,040
      elements — the reported number exactly. Container and engine innocent:
      st_init enforces span==numel·esz for float dtypes and load_t_n enforces
      numel==shared_inter·hidden, so the shard the engine loaded on Oct 22 is
      provably self-consistent; v2 only ever "worked" on fixtures whose permute
      target was the file's first tensor. Fix: hot_sort v2.1 — (begin,END)
      semantics, st.h-style checks at read time, st_init's duplicate-name rule
      (index.json resolves, else refusal), geometry pre-checks with fused-layout
      hints, BF16/U32/I64/F8 passthrough, `__metadata__` preserved, loud no-op
      warning (a silent no-op sort invalidates arm E). New `tools/st_dump.py`:
      read-only shard-header probe mirroring st.h's view (span checks, dup/index
      resolution, dir listing with sizes+mtimes) — the paste-back artifact for
      container mysteries. Gate: `scripts/test_hot_sort.py` (26 mechanical checks
      incl. the q27-layout regression) wired into `gates.yml`. Also corrected
      docs/BENCH_DENSE27B.md's "20 GB int4 container" claim — the pinned
      converter quantizes routed experts only; a dense 27B converts to ~50 GB
      F16 (the int4-trunk figures are the engine's load/map-time dialect).

- [x] **E11 — battery runner ran the STOCK dialect: every field arm OOM'd
      (CLOSED 2026-10-23).** Field report #3
      (`results/battery_llm-030_20261023-222216.json`, 7k prompt, MAXT guard
      passed): all seven compute arms rc=137 at an IDENTICAL ~5.10 GB
      fs-read watermark; E/F skipped cleanly (8149f4a's fix held). Cause —
      the runner invokes the binary directly but composed the launcher's
      dialect: no `TRUNK_MAP=1`, so the stock loader pulled the ~50 GB F16
      trunk into anon RAM and the 8 GB box killed it at its watermark (the
      nine arms re-proved BENCH_DENSE27B arm 1, "the stock attempt proves the
      OOM"), and `Q36_HOT_FRAC` is a launcher alias the binary never reads —
      engine takes `HOT_FRAC` (default 1.0). Same decoy in `run_phase5.sh`:
      every arm set `Q36_HOT_FRAC`/`Q36_GRIFFIN`, so the frac ladder A/B/C and
      the GRIFFIN A/B (D vs C) would have run vacuous — all arms frac 1.0,
      griffin off. No published phase-5 numbers tainted (only the manifest was
      committed; arms re-run with PHASE5_FORCE=1 after the fix — the resume
      cache would otherwise skip tainted .err files). Fix: both runners speak
      engine-native env (battery COMMON now mandates TRUNK_MAP=1 + HOT_FRAC +
      RAM_GB; new BATTERY_WARM — default 0/cold, the W read-matrix wants cold —
      and BATTERY_STREAM for E9 rotational-disk thrash; BATTERY_ARMS now
      actually selects arms instead of only filtering the summary); the battery
      JSON records the full operating point (engine path, frac, maxt, warm,
      stream, ram, model GB, meminfo) and per-arm RSS-at-kill + termination
      signal from /usr/bin/time -v, so rc=137 is self-diagnosing. Gate:
      `scripts/check_runner_env.py` — mechanical env-name oracle (every NAME=
      on an engine line of the runners must be getenv'd by engine/qwen36.c or
      the pinned colibri; ALLOW upstream claims re-verified; dense COMMON must
      carry TRUNK_MAP=1; decoys get an actionable message; negative test
      proves it catches the pre-fix lines). Loose mode in gates.yml; strict
      mode (`--colibri`, also scans q36-minipc.sh — its MoE CAP_RAISE/PIN/
      USAGE_SAVE verified colibri-core at bf24429) in engine-build.yml. The
      launcher itself was innocent: q36-minipc always mapped the aliases.

- [x] **E12 — the 3600-second wall: the post-E11 field battery proved the OOM
      fix and then every arm died at the runner's hardcoded timeout (runner
      fix CLOSED 2026-10-24; a valid field battery is still unmeasured).**
      Field runs #4/#5 (`results/battery_llm-030_20261024-114815.json` smoke
      arm A, `results/battery_llm-030_20261024-133054.json` full 9-arm,
      n_new=16; 13:30:54→20:31:21 ≈ 7 h ≈ 7 compute arms × 3600 s): E11's
      fix HELD — no rc=137 anywhere, peak RSS 6.95-7.0 GB under the 7.62 GB
      total, mem_available ~6.9 GB at end, no termination signals, operating
      point self-recorded (trunk_map=1, stream=1, frac 0.25). But every
      compute arm exited rc=124 at the runner's hardcoded `timeout 3600`
      with decode/ttft null: fs_read 37.1-41.9 GB/arm ≈ 0.75 × 54.7 GB =
      exactly one cold prefill pass at frac 0.25, i.e. an effective
      10.3-11.6 MB/s trunk-stream rate on the rotating disk — the arms were
      killed at the moment prefill was about to finish. The "~3-5 h
      overnight" guidance given with the E11 fix was ~10x off: the prefill
      floor alone is ~60-90 min/arm at this rate, and frac-0.25 decode
      re-reads GBs per token (the hot 13.7 GB cannot stay resident in a
      7.6 GB cache over a 54.7 GB model). Fix (mechanically tested with a
      fake-engine harness: rc=124 path, unlimited path, integer guard):
      `BATTERY_TIMEOUT` knob (per-arm seconds, default 3600, 0 = unlimited),
      per-arm logs now persist beside the JSON in `${OUT%.json}_logs/`
      instead of dying in /tmp (the 2026-10-24 progress lines — how far each
      arm got — were lost that way), an E12 I/O-budget preflight echo
      (cold-MB per pass → minutes at the measured rate), and the JSON
      records `timeout_s` + a note decoding rc=124 as "killed at the wall,
      not an engine fault". Docs: results/README field rows, BENCH_DENSE27B
      arm-1 addendum. Plan forward: probe ONE arm to completion
      (`BATTERY_ARMS="A" BATTERY_NNEW=2 BATTERY_TIMEOUT=0`), read its real
      TTFT + ms/token, then budget the matrix across nights; the 4b-slice-3
      window verdict stays blocked on completed W arms.

- [x] **E13 — the probe landed: FIRST COMPLETED FIELD ARM IN PROJECT HISTORY
      (rc=0). The daily point's measured floors: TTFT 61.3 min, decode
      37.1 min/token — and the matrix re-budget those two numbers force
      (field probe CLOSED 2026-10-25; the remaining arms are now SCHEDULED,
      not blocked).**
      The E12 probe ran exactly as prescribed (`BATTERY_ARMS="A"
      BATTERY_NNEW=2 BATTERY_TIMEOUT=0`, frac 0.25, maxt 8192, stream 1 →
      `results/battery_llm-030_20261025-101318.json` + its `_logs/`, the
      first field logs ever persisted): rc=0, wall 1:40:06, zero swaps,
      peak RSS 6.94 GB under the 7.62 GB total — E11's OOM fix holds on a
      COMPLETED run and E12's log persistence + unlimited-timeout path
      work end-to-end. The prompt is 1524 tokens ("7k" is characters):
      TTFT = 3676.22 s, landing inside E12's predicted 60-90 min prefill
      floor, at a steady ~50-53 s/layer (a 98-128 s disk transient at
      layers 47-51 shows in the progress lines and recovers). Decode =
      2,225,442 ms/token over the single independent decode step (N_NEW=2
      → token #1 rides out of the prefill forward; `[timers] decode: 1
      tokens`), and the phase split says where the 37 minutes go:
      moe/dense-MLP trunk 1596.6 s (71.7%), deltanet 411.4 s (proj 299.2 +
      norm/out 112.1; conv 9.7 MS, recurrence 18 MS), attention 127.4 s
      (core 17.2 s), lm_head 90.0 s (the 2.5 GB F16 vocab sweep). The
      compute cores are <1% of the token — decode at this operating point
      is pure weight streaming: fs_read 71.72 GB splits as ~6-7 GB load
      residency touch + ~41.0 GB prefill cold pass (0.75 × 54.7) + ~17-24
      GB decode-step re-read at ~8-11 MB/s, the E12 stream rate, with the
      ~6.9 GB resident window serving the rest of each 54.7 GB pass.
      Measured cost model (fits the wall time EXACTLY: 105 + 3676 + 2225 =
      6006 s): **T(arm) ≈ 105 s + 3676 s + (n_new−1) × 2225 s** → 1.7 h at
      n=2, 2.9 h at n=4, 5.4 h at n=8, 10.4 h at n=16 — the pre-probe
      default (9 arms × n=16) would have been ~4 days of continuous disk
      spinning. The generated text (`чисwik`, 2 greedy tokens) is the
      DOCUMENTED unsorted-container behavior — frac<1.0 on an unsorted
      container prunes arbitrary neurons, "mechanics valid, quality
      meaningless until hot_sort" (run_battery.sh header) — not a
      loader-fault signal; quality claims stay deferred to hot_sort
      (disk-space blocked) + phase-5 PPL. The spec-gate/lookup lines are
      degenerate (0/0 drafts, rho default) — one decode step leaves
      nothing to draft; cosmetic, not a fault. E/F skipped cleanly in the
      field for the first time ("skipped: no mtp head", rc=0).
      Re-budget: the remaining compute arms B/C/D + W0/W256/W1G at n_new=2
      cost 6 × ~1.7 h ≈ 10 h = ONE overnight run under BATTERY_TIMEOUT=0,
      and it completes the 4b-slice-3 window verdict (the W read-matrix
      wants exactly this: cold, frac 0.25, fs_read_gb per arm) plus field
      parity for the KV arms. Honest expectation for that night, stated up
      front: the trunk MLP stream (71.7% of every token) is arm-invariant,
      and at a 1524-token context the KV/window arms move at most the
      attention slice (~2.4 min of a 37-min token) — so this matrix
      measures MECHANICS and the W read-matrix on this host, not speed
      wins; the numbers will say so and that is a valid result, not a
      failed one. The only structural lever on the 37-min/token floor is
      trunk bytes per token: the container is NAMED q27_i4 but its dense
      trunk is F16 (the pinned converter quantizes routed experts only;
      the dense 27B has none), so a truly quantized trunk (i8 → 27.4 GB,
      i4 → 13.7 GB) would roughly halve/quarter the decode floor. That is
      offline tooling of the hot_sort class and blocked on the same
      disk-space wall (~42 GB free vs 50+ GB output) — recorded as the
      structural follow-up, deliberately NOT started in this slice.
- [x] **E14 — the overnight matrix landed: FIRST COMPLETE FIELD BATTERY
      (6/6 arms rc=0, 9 h 41 m unsupervised), the E13 cost model validated
      at −3.3%, fs_read proven arm-invariant across all seven completed
      arms, and the 4b-slice-3 window verdict: documented NEGATIVE — the
      window never allocated on the F16 trunk (field verdict CLOSED
      2026-10-25; Phase 4b fully closed with it).**
      The E13-prescribed overnight ran exactly as budgeted
      (`BATTERY_ARMS="B C D W0 W256 W1G" BATTERY_NNEW=2 BATTERY_TIMEOUT=0
      BATTERY_FRAC=0.25 BATTERY_MAXT=8192 BATTERY_STREAM=1` →
      `results/battery_llm-030_20261025-133007.json` + `_logs/`, 13:30:07 →
      23:11:05): every arm rc=0, zero swaps, engine PEAK RSS 6.94–6.96 GB
      under the 7.62 GB total — E11's OOM fix now holds on 7/7 completed
      field arms and E12's knobs (unlimited timeout + log persistence)
      survived their first unsupervised night. Wall per arm 5685–6067 s vs
      the E13 model's 6006 s: 10 h 00 m 36 s predicted vs 9 h 40 m 58 s
      measured, −3.3%, the residual being TTFT drift (arm mean 3431.5 s vs
      the probe's 3676.2 s — the probe had caught a 98–128 s disk transient
      at layers 47–51). The cost model is now field-validated and usable
      for phase-5 budgeting. Decode (`[timers] TOTAL`, one step each):
      2,234,889 / 2,275,205 / 2,264,507 / 2,357,981 / 2,322,810 /
      2,317,530 ms/tok for B/C/D/W0/W256/W1G, vs the probe's 2,225,442 →
      full seven-arm range 2225–2358 s = −2.6%/+3.2% about the 2285 s mean;
      TTFT 3287.6–3676.2 s across the seven = ±6% about 3466 s. No arm
      separates from any other beyond that
      noise band — exactly the honest expectation E13 pre-registered.
      fs_read 71.760–71.929 GB across the six (+ probe 71.72): a 0.21 GB
      (0.3%) total spread proves the trunk stream arm-invariant — KV codec
      (f32 → fp8-e4m3 → rot-int4 ±absorb) and the GRIFFIN/window knobs do
      not move read bytes at a 1524-token context; decode at the daily
      point is pure weight streaming, n=7. The three W arms agreeing to
      34 MB (0.05%) doubles as the repeatability floor of the /proc/io
      read accounting itself. B/C/D are the first COMPLETED lossy-KV field
      arms: banners verified engaged in-log (KV8 fp8-e4m3 + per-row scale;
      KV_TQ=4 codec=1 rot-int4 128 B/row, absorb 0 and 1), and all seven
      arms emitted the SAME first greedy token (`чисwik` — the documented
      unsorted-container pruning garbage; the argmax survives all three KV
      codecs, consistent with the cosine-band gates, quality still deferred
      to hot_sort + phase-5 PPL). RSS is codec-flat because KV at 1524
      tokens is ~100 MB-class against a ~7 GB working set: the KV tiers are
      long-context RAM-ceiling levers, not this host's trunk-bound decode
      levers — the field says what the synthetic batteries said.
      **The 4b-slice-3 gate-3 verdict (pre-registered: window-on must cut
      read bytes ≥50% at equal-or-better decode, else the knob stays
      default-off): FAIL, documented negative.** Measured cuts: W256
      −0.047%, W1G −0.018% vs W0. Two independent reasons, both recorded:
      (1) MECHANICAL — the window cannot allocate on q27_i4 at all:
      `q36_win_ensure` requires int4-packed MLP maps (`map_len == D*I/2`)
      and the dense trunk is F16 (`map_len == D*I*2`; every arm log re-proves
      it with `[dense-i8] 0 matrices quantized during load`), so `l->win`
      stays NULL and `q36_win_lookup` returns before touching the counters —
      the field signature `[qwen36 win] hits 0 misses 0 (0.0% hit-rate),
      budget 256/1024 MB` on both window arms. W0≡W256≡W1G were the same
      configuration three times; the flat matrix is an identity, not a
      window measurement. (2) ARITHMETIC — at the field-forced n_new=2
      there is ONE decode step: steady-state hits need ≥2 steps, and the
      decode re-read is 17–24 GB of the ~71.8 GB total (≤33%), so even a
      perfect window could not reach a ≥50% total cut. The threshold was
      written for multi-token decode before E12/E13 exposed the 37-min/token
      floor; calibration lesson recorded — future pre-registered gates must
      state their n_new. The knob-off verdict stands under either reading.
      First field positives on the way: the GRIFFIN=1 decode gather path
      ran to completion on the real container (3/3 rc=0, byte-identical
      greedy token, identical fs_read vs the non-GRIFFIN arms) — the
      slice-1 finalize fix works in the field. On the UNSORTED container at
      frac 0.25 the griffin set degenerates to the static hot prefix by
      construction (observe only sees the narrowed Ik = I·frac = 4352
      prefix neurons; top-k with k = I·frac can only re-select that same
      prefix), so the gather loop reads the identical contiguous rows
      per-row instead of batched — decode moe_total +5.0% on the W arms
      (1689–1717 s vs 1597–1638 s) is plausibly that per-row overhead, but
      at one decode step against a ±6% TTFT noise band it stays INCONCLUSIVE,
      recorded not claimed. A non-degenerate griffin set needs a sorted
      container (or frac 1.0) — the same disk-space wall as everything
      else quality-shaped. Phase 4b closes here: slice 1 mechanism +
      dense-observe fix, slice 2 synthetic sim curve
      (`results/results_window_sim_q36.json`), slice 3 field read-matrix
      negative. Window re-gate conditions (WINDOWING doc §Gates): a true
      int4-trunk container (E13's structural follow-up — also the ONLY lever
      on the 37-min/token floor, so the two waits are one wait), n_new ≥ ~8
      for steady-state hits, sorted container for non-degenerate sets; or
      the slice-2 plausible zone (MoE shared experts / smaller-I containers
      / bigger-RAM boxes). Next field step: phase 5 with `PHASE5_FORCE=1`,
      now budgetable from the validated cost model. This slice is
      docs-only (ROADMAP E14 + slice-3/CLOSED lines, results/README VALID
      matrix row, WINDOWING gate-3 verdict + slices, BENCH_DENSE27B arm-1
      addendum); no engine or runner change — the `hits 0 misses 0` line
      alone is ambiguous ("window unallocatable" vs "gather never ran"),
      but the code path resolves it: with GRIFFIN=1, dense observe +
      finalize-at-prefill-end build sets unconditionally (k = I·frac =
      4352 < I), so the gather DID run and the zero counters are the
      NULL-win short-circuit; had the trunk been int4, the same n_new=2
      run would have logged misses ≈ 4352×64 ≈ 278k and hits 0 — which is
      why the re-gate needs n_new ≥ ~8, not just a quantized trunk.
- [x] **E15 — Phase 5's first field attempt: three sessions died right after
      the manifest (`realweights_20261026-0919/0949/1014`, commit `cfc4e91` —
      manifest-only tombstones). The ref builder ran the engine in the STOCK
      dialect — the E11 failure mode, one level up.** `tools/run_phase5.sh`
      passed its `COMMON` (with `TRUNK_MAP=1`) only to `arm()`; the
      `make_wikitest_ref.py` call inherited none of it and the script's own
      env sets only SNAP/N_NEW/DUMP_IDS-class names — so the tokenizer's
      engine call hit `load_tq` without `TRUNK_MAP`: the stock path mallocs
      the whole 54.7 GB F16 dense trunk → dead in seconds on the 7.62 GB box,
      before any arm could start (no `arms/`, no `ref.json`, no `summary.json`
      in any of the three sessions; each retry re-hashed all 66 shards first,
      ~20 min per tombstone). `HOT_FRAC` unset would have defaulted to 1.0 on
      top of it. Second latent driver bug found in the same audit (E15b):
      M1 ran `PPL=1` + ref.json — the engine takes the tf_nll branch and
      returns BEFORE `generate()`, so `DUMP_IDS` is never written and M1
      could only ever print FAIL on missing files. Fixes this slice (all in
      `tools/run_phase5.sh` + `scripts/make_wikitest_ref.py`): WIKI-file
      fail-fast BEFORE the manifest hash; cached manifest on resume; ref
      build wrapped in `env TRUNK_MAP=1 HOT_FRAC=${PHASE5_REF_FRAC:-0.25}`
      with a right-sized slice (~6 bytes/token over ctx+len+128 — the ref ids
      are pure tokenizer output, `generate()` copies them verbatim before any
      forward, so the frac cannot taint ref.json, it only has to FIT);
      `--len=$((CTX+LEN))` (the old `--len=$LEN` silently scored LEN−CTX
      tokens against a "LEN scored" contract); `make_wikitest_ref.py`'s
      short-encode guard tightened from `ctx+64` to the full `--len` with an
      actionable message, plus a streaming-dialect hint on engine rc≠0; M1
      switched to `COMMON_GEN` (COMMON minus PPL=1) +
      `N_NEW=${PHASE5_M1NEW:-1}` = prefill + first-greedy-token bitwise
      determinism at the arm-A config, cached like the arms. Verified:
      run_phase5.sh end-to-end on a synthetic container (fresh + resume +
      M1 cached, 7/7 arms rc=0, summary.json well-formed) in-sandbox.
- [x] **E16 — Phase 5 was arithmetically unrunnable as scripted: `tf_nll`
      scored every token with a single-token `step()` — DECODE speed — and
      decode on llm-030 is 2225 s/token (E13, n=7). `PHASE5_LEN=2048` ⇒
      2048 × 37 min ≈ 52 DAYS per arm; ~9 runs per session ⇒ never.** The
      protocol's "2048 scored tokens — mini-PC budget" was priced at prefill
      speed; the field battery proved this box has no such speed for
      per-token forwards (every scored token re-streams the whole cold
      trunk: MLP trunk = 71.7% of the 2225 s step, lm_head ~90 s). Fix:
      `tools/patch_tf_chunk.py` → `Q36_TF_CHUNK=S` (S>1) scores S positions
      per weight pass through `step_ex`'s `nlogits` rows — the exact
      multi-row kernel speculative verify runs, whose decode-exactness the
      CI MTP forced-accept gates already prove byte-for-byte; unset/1 keeps
      the per-token path bit-identical (default preserved). Chain grows
      `... pretdw -tokdump_w-> pretfc -tf_chunk-> qwen36.c`; new CI gate
      (engine-build.yml) scores one ref at chunk 1/8/32, requires the
      `[tf] chunk=` banner and ppl within 1% of per-token. Sandbox proof
      this slice: patched engine vs pinned colibri bf24429, chunk 1/8/32 →
      TF-NLL identical to 4 dp (8.3178 nats, drift 0.0) on a synthetic
      container, banners + pass-counts correct (12 scored in 3 passes at
      chunk 8, 2 at chunk 32). What chunking does NOT fix: one lm_head pass
      per scored position (~90 s field) — the new per-token floor.
      Re-budget (E13 constants): per arm ≈ prefill(CTX) +
      ceil(LEN/CHUNK) × trunk pass + LEN × 90 s ⇒ defaults LEN=64 CHUNK=64
      ≈ 2.7 h/arm at frac 0.25, ~4.4 h at frac 1.0; full session (ref build
      + A B C D F G H + M1×2) ≈ 28–32 h continuous — a weekend, not 52
      days. Band consequence, pre-registered: at LEN=64 the ≤0.2% band (H)
      is UNDERPOWERED — paired same-chunk deltas keep B/C/F/G (1–5%)
      meaningful; H reports its measured Δ flagged underpowered (raise
      `PHASE5_LEN`, ~+1.6 h per extra 64 scored on frac-0.25 arms, if H
      must decide). Arm D note: the GRIFFIN gather engages only on S==1
      steps, so chunked scoring measures D as C-mechanics + finalize — on
      the UNSORTED field container D≡C was already the E14 finding
      (degenerate sets by construction); recorded, not a regression.

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
