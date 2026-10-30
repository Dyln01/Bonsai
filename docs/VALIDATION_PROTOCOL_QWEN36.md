# Phase 5 protocol — real-weights validation (sign-off document)

**Status:** awaiting user sign-off. **Nothing here downloads anything until the
checkbox at the bottom is ticked.** This document is the contract for what the
first (and only planned) real-checkpoint session does, measures, and accepts.
Every number it produces lands in `results/realweights_<date>/` and replaces
the synthetic projections in README *only* via `docs/REALWEIGHTS_QWEN36.md`.

## 0. Rules carried over from the whole project

- No real weights in the repo, ever (`.gitignore`: `*.safetensors *.gguf *.bin *.vhdx`).
- Greedy (`COLI_TEMP=0`) for every quality arm; decode numbers only from
  `[timers] TOTAL ms/token`; TTFT from the prefill timer sum (house convention).
- Every arm runs twice where determinism is claimed; identical `DUMP_IDS` or the
  arm is reported broken, not unlucky.
- Synthetic containers remain the regression floor: `gates.yml` + `engine-build.yml`
  must be green on the commit the real-weights session uses.

## 1. Checkpoints (choose at sign-off; both optional, either sufficient)

| id | model | class | source | approx size | license check at download |
|---|---|---|---|---|---|
| W1 | Qwen3.8-27B (dense) | dense 27B — the model this stack exists for | HF `Qwen/…` official | ~27B·2 B/int4-converted | record the repo's LICENSE text sha |
| W2 | Qwen3.6-35B-A3B | MoE + MTP head — exercises 2b end-to-end | HF official | ~35B converted | same |

Download outside the repo (`~/Models/`), record `sha256sum` of every shard in
`results/realweights_<date>/manifest.txt` before conversion. Convert with
colibri's own converter; keep the converter command line in the manifest.

## 2. Evaluation data (public, small, license-clean)

- **wikitest-raw:** wikitext-2 raw test split (CC-BY-SA), fetched with
  `scripts/fetch_wikitest2_raw.py ~/eval/wiki.test.raw` (the historical metamind
  S3 URL is dead — the script uses the `datasets` lib or the converted-parquet
  branch, and refuses to write suspicious payloads). Encoded with the container's real tokenizer into `ref.json`
  (`prompt_ids` + `full_ids`) so the engine's own `PPL=1` tf-nll path scores it.
- **agent-20:** 20 real multi-turn agent transcripts supplied by the user
  (privacy: stay on the mini-PC; only aggregate reuse % leaves the machine).
- **prompt-8k / prompt-long:** the existing synthetic prompts stay for
  mechanics arms (they are text, not weights).

## 3. Arms and pre-registered acceptance criteria

All arms on W1 unless noted; greedy; same wikitest ref.json; ppl via `PPL=1`.

| arm | config | acceptance vs A |
|---|---|---|
| A | stock-equivalent full compute: `Q36_HOT_FRAC=1.0 Q36_GRIFFIN=0 KV8=0` | baseline (also vs published ppl if known) |
| B | `Q36_HOT_FRAC=0.5` | Δppl ≤ +2% rel |
| C | `Q36_HOT_FRAC=0.25` | Δppl ≤ +5% rel (the exactness-dial claim) |
| D | C + `Q36_GRIFFIN=1` | Δppl ≤ C + 1% rel (per-prompt selection not worse than static) |
| E | C + sorted container (`calibrate`+`sort`) | Δppl ≤ C (sorting is relabeling: must be ~0) |
| F | C + `KV8=1` | Δppl ≤ +1% rel |
| G | C + `KV_TQ=4 KV_TQ_CODEC=1 Q36_TQ_ABSORB=1` | Δppl ≤ +2% rel |
| H | C + `Q36_FAST_SOFTMAX=1` | Δppl ≤ +0.2% rel |
| I | W2 + `Q36_MTP=1 Q36_MTP_DRAFTS=2` | report acceptance tok/forward; **no ppl regression vs W2 knob-off ≤ +0.5% rel** (verify exactness) |

Any arm outside its band is a **documented negative result**, not a tuned
rerun: the knob's default stays off and the number goes in the report as-is.

## 4. Mechanics + envelope arms (mini-PC, the claims that matter there)

- M1 determinism: arms A and I twice, `DUMP_IDS` identical.
- M2 prefix reuse: agent-20 through `serve` with `Q36_SLOTS` = conversations;
  reuse ≥ 80% on turns ≥ 2 (CI gate threshold, now on real tokens).
- M3 calibration: `calibrate` (bounded, `Q36_CAL_TOKENS=1024`) → top-25%
  energy coverage report from `hotcal_pipeline.py` ≥ 80% average.
- M4 envelope: peak RSS per arm ≤ 7.8 GB on the 8 GB box; decode tok/s table
  (this replaces README's projected rows).
- M5 MTP gate inputs: `tools/run_battery.sh` arms E/F on W2 → the gate's `d`
  and T(n) with real acceptance rates.

## 5. Session procedure (numbered, resumable)

1. `git pull` on the mini-PC; verify `verify_engine.sh` ALL PASS and CI green on HEAD.
2. Download + sha256 + convert (manifest.txt).
3. Build wikitest ref.json (tokenizer from the container; script added with the
   session commit: `scripts/make_wikitest_ref.py`).
4. Run arms A–I (ppl), then M1–M5; each arm's stderr to
   `results/realweights_<date>/<arm>.err`.
5. Write `docs/REALWEIGHTS_QWEN36.md`: table per arm vs acceptance, verdicts,
   negative results verbatim; update README headline table **from this file only**.
6. Commit results JSON + docs; tag `phase-5`.

## 6. Sign-off

- [x] I (the repo owner) approve downloading the checkpoint(s) marked in §1
      to `~/Models/` on the mini-PC for this one validation session, under the
      checkpoints' own licenses; nothing enters the repo.

**Signed in chat, 2026-10-20:** "continue with phase 5 skip phase 4 for now".
Scope recorded: **W1 (dense 27B) first**; W2 (MoE+MTP arms) on a follow-up
session whenever wanted. Session runner: `tools/run_phase5.sh` (resumable,
per-arm caching, bands mirrored from §3, summary JSON out).

## 7. Addendum (2026-10-26, E15/E16) — field runnability of §3–§5 on llm-030

The first field attempt (three manifest-only sessions,
`results/realweights_20261026-*`, ROADMAP E15/E16) exposed two
contract-level facts. The §3 bands are UNCHANGED; the mechanics are amended
as follows, implemented in `tools/run_phase5.sh` + `tools/patch_tf_chunk.py`
(CI parity gate in `engine-build.yml`):

- §5.3 ref build: runs the engine in the streaming dialect (`TRUNK_MAP=1` +
  `HOT_FRAC=${PHASE5_REF_FRAC:-0.25}`) over a right-sized wikitext slice
  (~6 bytes/token for ctx+len). The ref ids are pure tokenizer output —
  `generate()` copies the prompt ids verbatim before any forward — so the
  prefill config cannot taint them; it only has to fit the box (E15).
- §3 scoring: arms run `Q36_TF_CHUNK=${PHASE5_CHUNK:-64}` — chunked
  teacher forcing on the speculative-verify multi-row kernel (already
  CI-proven decode-exact). Per-token scoring is decode speed in the field:
  2225 s/token (E13) ⇒ 2048 scored ≈ 52 days/arm; chunking amortizes the
  trunk stream and leaves the lm_head pass (~90 s) as the per-scored-token
  floor (E16). `PHASE5_LEN` default: 2048 → 64 scored tokens. Budget:
  per arm ≈ prefill(CTX) + ceil(LEN/CHUNK) × trunk pass + LEN × 90 s ⇒
  ~2.7 h/arm at frac 0.25, ~4.4 h at frac 1.0, full session ≈ 28–32 h.
- Band power at LEN=64, pre-registered: B/C/F/G (1–5% bands) are paired
  same-chunk deltas and stay meaningful; the ≤0.2% band (H) is
  UNDERPOWERED — H reports its measured Δ flagged underpowered unless
  `PHASE5_LEN` is raised (~+1.6 h per extra 64 scored on frac-0.25 arms).
- Arm D note: the GRIFFIN gather engages only on S==1 steps, so chunked
  scoring measures D as C-mechanics + finalize; on the unsorted field
  container D≡C was already the E14 field finding (degenerate sets).
  Recorded as-is, not a regression.
- §4 M1: runs in GENERATE mode (`PPL=1` returns inside tf_nll before
  `generate()` — `DUMP_IDS` would never land, E15b): prefill +
  `PHASE5_M1NEW` (default 1) greedy tokens, bitwise, at the arm-A config,
  twice; cached like the arms (`PHASE5_FORCE=1` reruns). E18: cached only
  when BOTH runs completed (the engine's final `Speed:` line in both
  `A_det*.err`) — a runner kill mid-M1 reruns the pair; a completed
  mismatch stays cached and reported FAIL (a finding, not a tombstone).
  Field cost scales with M1NEW: each greedy step at frac 1.0 measured
  ~1930 s (T(1), `4700d3f`'s A_det logs), so `PHASE5_M1NEW=64` ≈ 13.9 h
  PER RUN (27.9 h for the pair — the 2026-10-30 session's biggest item);
  the default 1 costs ~1.2 h.
- §3 arms cache on completion evidence (E18): an arm `.err` without the
  runner's `rc=` line is a kill tombstone, not a result — it is cleared and
  rerun on resume. (The 2026-10-27 re-run lost A/B this way: tombstones
  skipped as "cached", summary shipped no-data for the frac references.)
- E17 (2026-10-27) — numeric-validity precondition: every field run before
  `patch_mapf16_bias.py` executed with a numerically dead mapped-F16 trunk
  (weights read at 2⁻¹¹²; tf ppl == vocab_size exactly). §3 bands are only
  meaningful on an engine at or after that patch; `run_phase5.sh` enforces
  this with a tripwire that aborts any session whose arm reports
  ppl == vocab_size (fresh or cached), and the E17 CI gate
  (mapped-vs-resident tf-NLL parity ≤0.1% on a real-headed F16-dense
  container) keeps the conversion honest going forward. The first field
  arms A (complete) and B (interrupted) of `realweights_20261026-1014` are
  VOID as measurements; `manifest.txt` and `ref.json` from that session are
  engine-independent and remain valid.
