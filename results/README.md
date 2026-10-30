# results/ — measured JSON manifest

Every number quoted in `docs/` traces to one of these files (honesty chain:
**doc table → driver script → JSON here → commit**). Synthetic arms use
same-geometry containers; the field-run section below carries the real-weights
llm-030 sessions. Parsed by CI on every push.

| JSON | Driver | Report | Notes |
|---|---|---|---|
| `results_std.json` | `scripts/run_workloads.py` (cold/steady/long profiles) | `docs/REPORT.md` | std synth container baseline |
| `results_kv_serve.json` | `scripts/exp_kv_serve.py` | `docs/REPORT.md` | KV pool at CTX=131072, SERVE mode |
| `results_kv_trunk.json` | `scripts/exp_kv_and_trunk.py` | `docs/REPORT.md` | dense-trunk offload A/B |
| `results_pin.json` | `scripts/exp_pin_recovery.py` | `docs/REPORT.md` | learned-pin residency recovery |
| `results_qwen_kv8.json` | `scripts/exp_qwen27_kv8.py`, `exp_qwen27_kv8b.py` | `docs/KV8_QWEN36.md`, `docs/LOWRAM_QWEN36_FINAL.md` | KV8 arms incl. decode version progression |
| `results_qwen_final.json` | `scripts/exp_qwen27_final.py` | `docs/LOWRAM_QWEN36_FINAL.md` | final low-RAM matrix |
| `bench_q36.json` | `q36-minipc bench` (launcher battery) | `docs/BENCH_DENSE27B.md` | phase-split honest battery |
| `results_kvtq_q36.json` | `scripts/exp_kvtq_q36.py` | `docs/KVTQ_QWEN36.md` | the 8-arm KV_TQ matrix (RSS ladder + decode prices) |
| `results_prefix_q36.json` | `scripts/exp_prefix_q36.py` | `docs/PREFIX_REUSE_QWEN36.md` | reuse % through mux slot swaps |
| `results_softmax_q36.json` | `scripts/exp_softmax_q36.py` | `docs/SOFTMAX_QWEN36.md` | 5/5 pairs A/B + microbench |

Convention: decode tok/s only from the engine's `[timers] TOTAL ms/token` line;
`Speed:` lines include prefill and are never quoted as decode. New batteries
append here + cite the JSON in the doc that uses them.

## Field runs (real weights — llm-030, 8 GB RAM / HDD mini-PC)

| File | Runner | Verdict |
|---|---|---|
| `battery_llm-030_20261022-192421.json` | run_battery.sh pre-`8149f4a` | INVALID — runner hardcoded MAXT 8192 vs a 15,960-token prompt; all arms rc=1. Kept as a negative result. |
| `battery_llm-030_20261022-193000.json` | run_battery.sh pre-`8149f4a` | INVALID — frac 1.0 + f32 KV at 16k exceeds the 8 GB envelope by arithmetic; all arms rc=137. |
| `battery_llm-030_20261023-222216.json` | run_battery.sh pre-E11 | INVALID — runner launched the STOCK dialect (no `TRUNK_MAP=1`; `Q36_HOT_FRAC` alias inert): seven arms rc=137 at an identical ~5.10 GB fs-read watermark = arm 1 of `docs/BENCH_DENSE27B.md` re-proved nine times (ROADMAP E11). E/F skipped cleanly. |
| `battery_llm-030_20261024-114815.json` | run_battery.sh post-E11 | INVALID for numbers — smoke arm A died rc=124 at the hardcoded 3600 s wall mid-prefill (ROADMAP E12). VALID as the E11 field proof: `TRUNK_MAP=1`+stream, peak RSS 7.0 GB < 7.62 GB total, no signal-9 kill. |
| `battery_llm-030_20261024-133054.json` | run_battery.sh post-E11 | INVALID for numbers — 7 compute arms each ran the full 3600 s and were killed (fs_read 37-42 GB ≈ one cold pass at frac 0.25 → 10.3-11.6 MB/s effective); E/F skipped cleanly. The measured HDD budget is now in the runner (`BATTERY_TIMEOUT`, budget echo, `_logs/` persistence). |
| `battery_llm-030_20261025-101318.json` | run_battery.sh post-E12 | **VALID — the first completed field arm in project history** (E12 probe, arm A, n_new=2, frac 0.25, maxt 8192, stream 1, timeout unlimited): rc=0, wall 1:40:06, zero swaps, peak RSS 6.94/7.62 GB, TTFT 3676.22 s (1524-token prompt), decode 2,225,442 ms/token (`[timers] TOTAL`, one decode step; MLP trunk = 71.7%, attention core 17.2 s), fs_read 71.72 GB. Generated `чисwik` — ORIGINALLY read as unsorted-container pruning; **E17 CORRECTION: it was the bias-less mapped-F16 converter** (every trunk weight read at 2⁻¹¹² → residuals frozen at the embedding, decode = argmax of embed→lm_head alone). Timing/IO/RSS numbers in this row are UNAFFECTED (same bytes streamed, same FLOPs) and stay VALID. E/F clean field skips. Cost model: T(arm) ≈ 105 s + 3676 s + (n_new−1)×2225 s (ROADMAP E13). |
| `battery_llm-030_20261025-133007.json` | run_battery.sh post-E12 | **VALID — the first COMPLETE field matrix** (E13-prescribed overnight: B/C/D + W0/W256/W1G, n_new=2, frac 0.25, maxt 8192, stream 1, timeout unlimited; 13:30:07→23:11:05 = 9 h 41 m unsupervised): 6/6 rc=0, zero swaps, peak RSS 6.94–6.96/7.62 GB, E/F clean skips. Decode 2,234,889–2,357,981 ms/tok and TTFT 3287.6–3660.9 s — with the probe, all seven arms inside a ±6% HDD noise band, no arm separation (E13's pre-registered honest expectation). fs_read 71.760–71.929 GB on every arm (seven-arm spread 0.21 GB = 0.3%): the trunk stream is arm-invariant — KV codecs (f32/KV8/TQ4±absorb) and GRIFFIN/window move no bytes at 1524-token context (E17 note: the byte-level invariance stands as an IO fact; the OUTPUT-level identity across arms — same first greedy token, same `чисwik` — was the dead-trunk bug making every arm's COMPUTATION identical, not lossy-KV argmax parity). Cost model validated: 10 h 01 m predicted vs 9 h 41 m measured (−3.3%). **4b-slice-3 gate-3 verdict: FAIL, documented negative** — window never allocated (int4-only guard vs the F16 trunk; `hits 0 misses 0` on W256/W1G is that signature), `Q36_WIN_MB` stays 0; all seven arms emit the same first greedy token — E14 read this as lossy-KV argmax parity; **E17 correction: the numerically dead trunk made every arm's forward identical by construction**; the claim is VOID, the structural gate-3 FAIL stands (window never allocates on an F16 trunk regardless of numerics). ROADMAP E14+E17, WINDOWING §Gates. |
| `battery_llm-030_20261027-132202.json` | run_battery.sh, STALE engine | INVALID ×2 (E18) — (1) it ran 13:22→20:22 CET on Oct 27, hours BEFORE the E17 fix was pushed (`538981b` @ 21:23 UTC = 22:23 CET) and before the box rebuilt: the engine was still the DEAD one — fingerprint: every arm's output `чис`, the byte-identical prefix of E14's dead `чисwik`; (2) `BATTERY_NNEW=16` met the default `BATTERY_TIMEOUT=3600` on a box needing ~40,000 s/arm (TTFT ~3400 s + 16 × ~2300 s decode) → 7/7 arms rc=124 at the wall. 7 h burned; only the kill-point envelope is quotable (fs_read 39.5–41.7 GB mid-prefill, peak RSS 6.96–6.99/7.62 GB, mem_available 6.92 GB at end). `20261025-133007` REMAINS the valid timing matrix (timing was never invalidated by E17). E18b guard now hard-blocks the timeout-vs-NNEW mismatch (rc=2; `BATTERY_FORCE_SHORT_TIMEOUT=1` overrides for fast models). A live-numerics battery rerun (for decode TEXT under the fixed engine) is optional and waits its own ~10 h night: `BATTERY_NNEW=2 BATTERY_TIMEOUT=0 BATTERY_FRAC=0.25 BATTERY_MAXT=8192 BATTERY_STREAM=1`. |
| `realweights_20261022-2011/manifest.txt` | run_phase5.sh preflight | VALID — 66 shard sha256s of `/root/Models/q27_i4` (conversion provenance). Only the preflight manifest was committed; any pre-E11 arm `.err` on the box is tainted (decoy `Q36_*` aliases → every arm ran frac 1.0/griffin 0) — re-run with `PHASE5_FORCE=1`. |
| `realweights_20261026-0919/manifest.txt` (+ `-0949/`, `-1014/`) | run_phase5.sh pre-E15 | INVALID — preflight tombstones ×3 (commit `cfc4e91`): the ref builder ran the engine WITHOUT `TRUNK_MAP=1`/`HOT_FRAC` — the stock loader mallocs the 54.7 GB F16 trunk and the 7.62 GB box kills it in seconds; no `ref.json`, no `arms/`, no `summary.json` (ROADMAP E15). The manifests themselves are VALID provenance — identical 66 sha256s ×3, a free three-way repeatability proof of the container. `-1014/` is the resume target of the fixed re-run (E15+E16 kit: streaming-dialect ref build, chunked tf-nll `Q36_TF_CHUNK`, LEN default 64, session budget ~28–32 h). 2026-10-27 (`3f41ca8`): ref.json + arm A landed complete, arm B interrupted mid-run — arm A returned `ppl = 248320.00` = vocab_size EXACTLY, the E17 signature (mapped-F16 weights at 2⁻¹¹² → uniform logits): **A/B VOID, session halted**. `ref.json` + `manifest.txt` stay valid (tokenizer/sha256 only — engine-independent). Re-run: rebuilt engine + `rm -f arms/*` + same command. 2026-10-30 (`4700d3f`): the post-fix re-run PARTIALLY landed — **C/D/F/G/H VALID, the first live-numerics field quality numbers** (512 ctx / 64 scored, chunk 64): C (frac 0.25) tf-NLL 7.4907 nats/tok, ppl 1791.28, wall 10091.3 s = cost model +3.8%, RSS 7.01/7.62 GB; D ≡ C EXACT (window knob inactive — gate-3 negative persisting live); H ≡ C EXACT (FAST_SOFTMAX); F (KV8) +0.0517 nats (+5.31% ppl vs C); G (TQ4-absorb) +0.1209 nats (+12.85% ppl vs C). **M1 PASS** at `PHASE5_M1NEW=64`: 576-id streams byte-identical ×2 at frac 1.0 (27.9 h) — carrying the first live spec-gate field numbers (T(1) ≈ 1930–1936 s, acceptance 82.0%, 2.86 tok/forward). **A/B are E18 tombstones** (runner killed early in load — `.err` without `rc=`, empty `.out` — then skipped as "cached" by every resume): the summary's `no-data`/`FAIL` verdicts are missing-reference artifacts, NOT quality verdicts; all §3 bands wait on A. E18-fixed runner self-heals tombstones: the ~6 h resume (A+B only; C..H/M1/manifest/ref cached) completes `summary.json`. |

Kept per house rules: negative results documented, not buried. As of
2026-10-25 the field battery is COMPLETE and valid (probe + the 6-arm
overnight matrix above — quoted in ROADMAP E13/E14, the
`docs/BENCH_DENSE27B.md` arm-1 addendum, and the `docs/WINDOWING_QWEN36.md`
gate-3 verdict, all citing those two JSONs). Phase 4b is closed by it.
2026-10-26: the first phase-5 attempt landed as three manifest-only
tombstones (E15: stock-dialect ref build; E16: per-token tf-nll = ~52
days/arm at LEN 2048). The kit was fixed and the scoring chunked
(`Q36_TF_CHUNK`, CI-parity-gated) the same day; the re-run needs NO
`PHASE5_FORCE` (no arm ever ran) and resumes
`results/realweights_20261026-1014` on its cached manifest. Its
`summary.json` is the remaining field artifact; it appends here when it
lands.
2026-10-27: the re-run's first completed arm (A, frac 1.0) exposed E17 — the
mapped-F16 exponent-bias bug that made EVERY prior field forward numerically
dead (tf ppl == vocab_size exactly; `чисwik` and the arms' identical outputs
reinterpreted in the rows above; timing/IO results unaffected). Engine fixed
(`patch_mapf16_bias.py`), E17 CI gate added (mapped-vs-resident parity on a
real-headed F16-dense container), and `run_phase5.sh` now aborts on any arm
reporting ppl == vocab_size. The session re-runs from `-1014/` with cleared
`arms/` on the rebuilt engine; its `summary.json` remains the field artifact
this table waits for.
2026-10-30: the post-E17 re-run landed (`4700d3f`) — C/D/F/G/H + M1 are LIVE
and valid: the first real quality numbers of the field checkpoint (tf-NLL
7.4907 nats/tok at frac 0.25; knob deltas structured — D≡H≡C exact, KV8
+0.0517 nats, TQ4-absorb +0.1209 nats; M1's 576-id streams byte-identical
twice at frac 1.0, with the first live spec-gate numbers inside: 82.0%
acceptance, 2.86 tok/forward) — but arms A/B fell to runner-kill tombstones
that the old resume cache skipped (E18), so the summary shipped no-data for
both frac references and every §3 band is pending. The Oct-27 battery rerun
was pre-fix AND timeout-walled (invalid ×2, row above). E18 hardening:
completion-evidence caching (`rc=` line for arms, final `Speed:` line for
M1), battery timeout-vs-NNEW guard, exec bit restored. Remaining field step:
a ~6 h resume fills A/B; `docs/REALWEIGHTS_QWEN36.md`, the README headline
de-projection and the `phase-5` tag all follow that completed
`summary.json`.
