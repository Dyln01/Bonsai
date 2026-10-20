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

- **wikitest-raw:** wikitext-2 `wiki.test.raw` (~2 MB, CC-BY-SA): the perplexity
  corpus. Encoded with the container's real tokenizer into `ref.json`
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

- [ ] I (the repo owner) approve downloading the checkpoint(s) marked in §1
      to `~/Models/` on the mini-PC for this one validation session, under the
      checkpoints' own licenses; nothing enters the repo.

Date / initials: ______________________
