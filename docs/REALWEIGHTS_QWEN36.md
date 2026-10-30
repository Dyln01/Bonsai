# REALWEIGHTS_QWEN36.md — Phase 5 W1 session report (real-weights validation)

**Session:** `results/realweights_20261026-1014` · **completed 2026-10-30** (`summary.json`
ts 2026-10-30T16:42:30, uploaded `d93b7a6`) · **tag:** `phase-5`
**Host:** llm-030 — 4-core mini-PC, 7.62 GB RAM, rotating disk (~10–12 MB/s measured)
**Container:** `/root/Models/q27_i4` — Qwen3.8-27B-class dense (W1), 64 layers
(16 attn + 48 DeltaNet), hidden 5120, shared_inter 17408, vocab 248320, **F16 dense
trunk (54.7 GB)** in colibri dialect; 66 shards sha256'd in `manifest.txt` before
conversion (protocol §1). No weights or shards are in this repo, ever.
**Engine:** colibri `qwen36.c` @ pinned `bf24429` + the 25-hop Bonsai patch chain,
FINAL md5 `bf08d9aa9b8194a408a94b13bf16a70b` — **post-E17** (the mapped-F16
exponent-bias fix, `538981b`); `verify_engine.sh` ALL PASS and CI green on the
commits the session ran under. The runner tripwire (abort on `ppl == vocab_size`)
never fired on the re-run: every arm's ppl is far from 248320.
**Runner:** `tools/run_phase5.sh` post-E18 (`0d462c4`) — completion-evidence
caching; the A/B resume self-healed the two tombstones exactly as designed.
**Protocol:** `docs/VALIDATION_PROTOCOL_QWEN36.md` (signed 2026-10-20, W1
dense-first). Scoring: wikitext-2-raw test slice, ctx 512 / 64 scored tokens,
`Q36_TF_CHUNK=64` (E16, CI-parity-gated vs per-token), greedy, `ref.json`
(512 prompt_ids + 576 full_ids) built by the container's own tokenizer.

This file is the sole source for the README headline update (§5.5). Every number
below traces to a committed artifact under `results/realweights_20261026-1014/`.

---

## 1. Headline verdict (plain language)

**The full-size dense 27B ran, end to end, on the 8 GB box — and its first honest
quality number is good: arm A (full compute, `HOT_FRAC=1.0`) scored tf-NLL 1.9076
nats/token, ppl 6.74 on wikitext-2-raw at 512 ctx / 64 scored tokens, rc=0, peak
RSS 6.96/7.62 GB.** The same arm on the same container returned ppl 248320.00
(vocab-size exactly) on the dead pre-E17 engine on Oct 27 — the field itself is
the strongest possible confirmation that the E17 fix restored the trunk to the
numerics.

**The compression dial is a documented negative on this container as-shipped.**
`HOT_FRAC` 0.5 and 0.25 degrade ppl to 29.24 and 1791.28 — because the dial's
contract is a *hotness-sorted* container (top-energy neurons in the file prefix),
and the field container is **unsorted**: the sort step (arm E) was disk-blocked
(~42 GB free vs ~50 GB sorted output), so frac<1.0 selected an arbitrary neuron
subset — random MLP dropout, with the measured dose-response to match. The
sorted-container case (arm E, the mechanism the claim was written for) remains
**pending**, not refuted. Until E lands, the honest field setting is
`HOT_FRAC=1.0`: exact full compute, ~2000 s/token decode off HDD.

Lossy-KV knobs also came back negative at this window (KV8 +5.31% ppl,
TQ4-absorb +12.85%, both over their pre-registered paired budgets) — they stay
default-off. Determinism (M1) and the RAM envelope (M4) PASSED, and the
self-speculative machinery produced its first live field numbers: **82.0%
acceptance, 2.86 tokens/forward**.

## 2. §3 quality arms — measured vs pre-registered acceptance

Verbatim from `summary.json` (runner-generated; bands mirrored from §3, never
moved). `dRel(A)` is the runner's literal vs-A reading; the paired reading is
reported alongside per the §7-addendum pre-registration ("H reports both ways";
D/E judged vs C). Δnats are exact differences of the printed tf-NLL means.

| arm | config | tf-NLL nats/tok | ppl | wall (s) | RSS GB | dRel(A) | band (§3, pre-registered) | verdict |
|---|---|---|---|---|---|---|---|---|
| **A** | frac 1.0, full compute | **1.9076** | **6.74** | 9888.0 | 6.96 | — | baseline ("also vs published ppl if known" — none recorded for this checkpoint) | **the reference** |
| **B** | frac 0.5 | 3.3755 | 29.24 | 10603.8 | 6.87 | +333.83% | ≤ +2% vs A | **FAIL — documented negative** |
| **C** | frac 0.25 | 7.4907 | 1791.28 | 10091.3 | 7.01 | +26476.85% | ≤ +5% vs A (the exactness-dial claim) | **FAIL — documented negative** |
| **D** | C + GRIFFIN | 7.4907 | 1791.28 | 10058.4 | 6.89 | (+26476.85%) | ≤ C + 1% | **PASS vs C: Δ = 0.000%** (identical to 4 dp — knob output-transparent here) |
| **E** | C + sorted container | — | — | — | — | — | ≤ C | **NOT RUN — disk-blocked** (see §3) |
| **F** | C + KV8 (e4m3 KV) | 7.5424 | 1886.39 | 7631.4 | 6.83 | +27888.0% | ≤ +1% | **FAIL on both readings** — paired vs C: +5.31% ppl (+0.0517 nats), 5.3× over budget |
| **G** | C + KV_TQ4 absorb | 7.6116 | 2021.48 | 8218.4 | 6.65 | +29892.3% | ≤ +2% | **FAIL on both readings** — paired vs C: +12.85% ppl (+0.1209 nats), 6.4× over budget |
| **H** | C + FAST_SOFTMAX | 7.4907 | 1791.28 | 10443.7 | 6.70 | +26476.85% | ≤ +0.2% | literal vs-A **FAIL is a pairing artifact** (inherits C's collapse); **paired vs C: Δ = 0.000% — within band** (flagged underpowered at 64 scored, §7 addendum) |
| **I** | W2 + MTP drafts | — | — | — | — | — | n/a | **NOT RUN** — W2 (MoE+MTP) is the follow-up session per sign-off scope |

All seven run arms: rc=0, `[tf] chunk=64: 64 scored tokens in 2 weight passes`
banner present, 64/64 tokens scored. The session's only two runner kills hit the
first A/B attempts (the E18 tombstones); the E18-fixed resume healed both and no
arm needed a manual `rm`.

Reading notes, honest and complete:

- **A = 6.74 ppl is a single 64-token window** of wikitext-2-raw (one 512-token
  prompt from the test slice). It is non-degenerate by a factor of ~37,000 vs the
  uniform-softmax value (ln V = 12.42 nats / ppl 248320), monotone-consistent
  with the frac ladder, and in the range a healthy 27B-class dense model sits at
  on raw wikitext at short context — but 64 scored tokens is a narrow sample;
  treat it as *the* first field reference point, not a leaderboard number. No
  published wikitext-2-raw ppl for this checkpoint is recorded in the repo, so
  the §3 "vs published" clause is reported empty, as found.
- **The frac ladder is monotone and mechanistically clean**: 1.9076 → 3.3755 →
  7.4907 nats at frac 1.0 → 0.5 → 0.25. That is the dose-response of dropping
  0% → 50% → 75% of the shared-MLP neuron rows (gate/up rows + down columns
  `[0, Ik)`, `Ik = ⌊17408·frac⌋`, engine `qwen_shared_experts_cpu`) chosen by
  *file order* — on an unsorted container, an arbitrary subset.
- **D ≡ C and H ≡ C to all printed digits** (7.4907): GRIFFIN's gather engages
  only on S==1 decode steps, so chunked tf-scoring measures D as C-mechanics
  (E14's finding persists in live numerics); FAST_SOFTMAX is output-transparent
  at this window. Both are *knob-inert* results, not quality endorsements.
- **F/G negatives are real on the paired reading** — the only reading that
  isolates the KV codec's own cost: +0.0517 and +0.1209 nats over C, 5–6× their
  whole budgets. Single window, 64 tokens; pre-registration says report as-is,
  no tuned rerun. **KV8 and KV_TQ4-absorb stay default-off at this checkpoint.**
- **H's literal vs-A FAIL** is the §3 pairing artifact flagged in advance: H's
  config *is* C + one output-transparent knob, so its vs-A delta is C's delta.
  Both readings are on the record; neither changes a knob default (FAST_SOFTMAX
  was already opt-in).

## 3. Arm E — why the exactness dial's own mechanism is still pending

The claim under §3-C ("Δppl ≤ +5% — the exactness dial") was written against the
HOT-NEURON contract (`docs/HOTNEURON_DENSE27B.md`): neurons **sorted by
calibrated activation importance**, the hot prefix = the top-energy fraction,
synthetic gate rel-err < 5e-5 *on the sorted container*. The field container was
never sorted: `calibrate`+`sort` need ~50 GB for the sorted output and the box
had ~42 GB free, so arm E skipped cleanly (runner logs the skip) and arms B/C/D
ran the dial **on an unsorted container** — where the prefix is file order, i.e.
an arbitrary 50%/25% of neurons. The FAILs above are therefore attributed, in
the pre-registered record: **the dial measured what it was given — random MLP
neuron dropout — and the numbers are kept as-is.** What they do NOT test is the
sorted-prefix mechanism itself; that is arm E, and it is the first thing to run
when disk allows (free/attach ~60 GB, then
`PHASE5_M3=1 tools/run_phase5.sh /root/Models/q27_i4` reruns calibrate+sort+E
only; everything else stays cached).

An int4-packed trunk would also change the physics twice over: the shared-MLP
prefix at frac 0.25 would be ~2.1 GB instead of this F16 trunk's ~8.6 GB (which
alone exceeds the box's entire 7.62 GB RAM) — a page-cache-resident working set,
the warm-decode regime the README's original 12–14 tok/s projection assumed and
this F16 container could never reach (see §4 speed table). That is a container
change, not a knob; it is listed in §8, not measured here.

## 4. §4 mechanics arms

- **M1 determinism — PASS.** Arm-A config (frac 1.0), generate mode,
  `PHASE5_M1NEW=64`: prefill(512) + 64 greedy steps, run twice;
  `A.ids` == `A2.ids` **byte-identical** (576 ids, 2632 B each; re-verified with
  `cmp` on the committed files). Walls 49750.9 s + 50370.5 s = 100,121 s
  (27.8 h) — the session's single biggest budget item, run at the user's
  explicit M1NEW=64.
  Inside those two logs, the **first live spec-gate field numbers**: T(1)
  1,935,873.9 / 1,929,637.8 ms at frac 1.0; verify-row scaling 2:1.04 3:1.09
  4:1.15 5:1.20 6:1.24 (rho 0.06–0.07); acceptance-by-position
  0.80/0.93/0.90/0.83/0.75; lookup **2.86 tokens/forward** (22 forwards per 63
  tokens), **acceptance 82.0%** (41/50 drafts in 18 verifies), 3 declined, 5
  probes; TTFT(512 @ frac 1.0) **2460.54 s**. The wall arithmetic closes:
  22 forwards × ~2000–2300 s + TTFT ≈ 49.7 ks.
- **M2 prefix reuse (agent-20 through `serve`) — NOT RUN.** Needs the user's 20
  transcripts on the box and a serve session; deferred, listed in §8. (The
  synthetic 93% reuse result stands in `docs/PREFIX_REUSE_QWEN36.md`, unchanged.)
- **M3 calibration coverage — NOT RUN.** Same disk wall as arm E (calibrate
  output feeds sort). Runs with E.
- **M4 envelope — PASS on RSS; speed table below replaces the README
  projections.** Peak RSS across all seven arms: **6.65–7.01 GB ≤ 7.8 GB** on
  the 7.62 GB box (A 6.96, B 6.87, C 7.01, D 6.89, F 6.83, G 6.65, H 6.70;
  M1 det runs 6.76). No arm was OOM-killed; no swap storm (zero swaps in the
  battery logs). The honest envelope is ~7 GB, **not** the ~4 GB the synthetic
  projection carried — the field container's embed/head/vision/DeltaNet
  residents are bigger than the student analog's.

  Field speed table (house conventions: decode only from `[timers] TOTAL
  ms/token`; `Speed:` lines are wall-clock summaries, never decode):

  | quantity | measured | source |
  |---|---|---|
  | decode, frac 0.25, 1524-ctx | **2,225–2,358 s/token** (`[timers] TOTAL`) | battery `20261025-*` (E13/E14, VALID — timing was never touched by E17) |
  | decode step, frac 1.0, 512-ctx | T(1) **1,936 / 1,930 s** (spec-gate first-row timing; det logs carry no `[timers]` line) | `arms/A_det*.err` |
  | TTFT, 512 tok @ frac 1.0 | **2460.5 s** | `arms/A_det.err` |
  | TTFT, 1524 tok @ frac 0.25 | 3287.6–3676.2 s | battery `20261025-*` |
  | tf-arm walls | 2.12–2.95 h per 64-scored arm (F 7631 s … B 10604 s; ±25% HDD luck between arms — quality numbers are wall-independent) | `arms/*.out` |

  The README's projected "12–14 tok/s warm" row is **de-projected by this
  table**: warm residency needs the per-token working set to stay page-cache
  resident, and this F16 trunk's frac-0.25 shared-MLP prefix alone is ~8.6 GB
  (64 layers × 3 matrices × 5120×17408 × 2 B × 0.25) — more than the box's
  entire 7.62 GB — with the attention/DeltaNet weights read in full on top.
  Every token streams off a ~10–12 MB/s disk. The projection was arithmetic for
  an int4 trunk (~2.1 GB prefix, the HOT-NEURON doc's own "~2 GB working set");
  it was never achievable on this container. Measured field decode is
  ~0.0005 tok/s.
- **M5 MTP gate inputs — NOT RUN.** Requires W2 (MoE + MTP head container),
  follow-up session per sign-off scope.

## 5. Cost model — third field validation

Predicted (E16 model, E13-validated constants): per arm ≈ prefill(CTX) +
ceil(LEN/CHUNK) × trunk pass + LEN × ~90 s lm_head floor; A/B budgeted ≈ 2.9–3.0 h
each for the resume. Measured: **A 2.75 h, B 2.95 h** — inside the budget band;
A's internals decompose exactly: prefill passes 2422 s + 1852 s = 4274 s,
remainder 5614 s ≈ 64 × 87.7 s lm_head. C had landed at +3.8% of the model.
Three independent field confirmations (battery −3.3%, C +3.8%, A/B within
−8…+2% of the budgeted range): the model is the planning instrument it was
built to be.

## 6. Session accounting (what the ~5 days bought)

Oct 26: three preflight tombstones (E15 stock-dialect ref build — fixed same
day; manifests VALID provenance, identical 66 sha256s ×3). Oct 27: first arm A
complete → ppl == vocab_size exactly → **E17** (bias-less `map_f16_one`: the
trunk had been numerically absent from every field forward ever run); engine
fixed + CI-gated + runner tripwired; all pre-fix quality readings void, all
timing/IO/RSS readings unaffected. Oct 27 battery rerun: INVALID ×2 (stale dead
engine + timeout-vs-NNEW wall — E18b guard now blocks it). Oct 28–30: post-fix
re-run banks C/D/F/G/H (~12.9 h) + M1 (27.8 h); runner killed mid-A/B →
tombstones skipped by the old cache → **E18** (completion-evidence caching).
Oct 30: E18-fixed resume lands A + B in ~5.7 h, self-healing both tombstones
with no manual `rm`; `summary.json` final at 16:42. Valid engine time ≈ 46 h
(C–H 12.9 h + M1 27.8 h + A/B 5.7 h), on top of the voided Oct-27 arm A
(~2.9 h) and the three preflights — on a box that spent its first three field
days proving, and then surviving, four defects of its own validation kit
(E15–E18). Every one is in the ROADMAP ledger with its fix, gate, and
field-failure reproduction.

## 7. Documented negatives (pre-registered disposition, verbatim)

Per §3: "Any arm outside its band is a documented negative result, not a tuned
rerun: the knob's default stays off and the number goes in the report as-is."

1. **B — `HOT_FRAC=0.5` on the unsorted field container:** ppl 29.24 vs A 6.74
   (+333.83%, band +2%). Default stays 1.0 on unsorted containers.
2. **C — `HOT_FRAC=0.25` on the unsorted field container:** ppl 1791.28
   (+26476.85%, band +5%). The exactness-dial claim is NOT validated by this
   run — it was tested outside its contract (unsorted prefix); its own test
   (arm E, sorted) is pending on disk space. Default stays 1.0 until E lands.
3. **F — KV8 at this checkpoint/window:** +5.31% ppl vs C (+0.0517 nats, band
   +1% paired). KV8 stays off for quality-critical runs; its −26% RSS lever
   (synthetic, 8k ctx) remains available as a labeled RAM-vs-quality trade.
4. **G — KV_TQ4-absorb at this checkpoint/window:** +12.85% ppl vs C (+0.1209
   nats, band +2% paired). Stays off.
5. **H — literal vs-A band FAIL (+26476.85%)** recorded as the pre-flagged
   pairing artifact; paired reading Δ = 0.000% (underpowered at 64 scored).
   FAST_SOFTMAX remains opt-in, output-transparent at this window.
6. **E, I, M2, M3, M5 — not run** (disk / W2-scope), listed as pending, not
   failed.

## 8. What remains (next field steps, in order)

1. **Arm E + M3** when ~60 GB is free (external disk or artifact cleanup):
   `PHASE5_M3=1 tools/run_phase5.sh /root/Models/q27_i4` — calibrate (coverage
   report ≥ 80% per §4-M3) → sort → arm E at frac 0.25, band ≤ C. This is the
   sorted-prefix mechanism's first real-weights test; everything else in the
   session stays cached. Budget ≈ calibrate + sort IO + one 2.8 h arm.
2. **M2** (agent-20 prefix reuse through `serve`) — needs the transcripts on
   the box; reuse ≥ 80% on turns ≥ 2 is the pre-registered bar.
3. **W2 session** (MoE + MTP): arm I + M5, whenever wanted (sign-off scope).
4. **Optional** live-numerics battery rerun (~10 h night,
   `BATTERY_NNEW=2 BATTERY_TIMEOUT=0 BATTERY_FRAC=0.25 BATTERY_MAXT=8192
   BATTERY_STREAM=1`) for decode TEXT under the fixed engine — not required for
   phase-5 sign-off.
5. **Container experiment** (outside phase-5 scope): an int4-packed trunk of
   the same checkpoint would make the frac-0.25 shared-MLP prefix (~2.1 GB)
   page-cache resident — the warm-decode regime — and is the only measured path from
   ~2000 s/token toward the tok/s range the student analog showed. New
   manifest + new session if pursued.

## 9. Artifact index (everything this report quotes)

| file | proves |
|---|---|
| `manifest.txt` | 66 shard sha256s — conversion provenance, three-way identical across the Oct-26 preflights |
| `ref.json` | wikitext-2-raw slice encoded by the container's tokenizer: 512 prompt_ids / 576 full_ids |
| `arms/A.err`, `A.out` | frac-1.0 baseline: full boot log, both prefill passes, chunk banner, rc=0, 1.9076 nats, ppl 6.74, wall 9888.0 s, RSS 6.96 |
| `arms/B.err`, `B.out` | frac-0.5: rc=0, 3.3755 nats, ppl 29.24, wall 10603.8 s, RSS 6.87 |
| `arms/{C,D,F,G,H}.{err,out}` | frac-0.25 family: nats/ppl/wall/RSS per §2 table, all rc=0 with chunk banners |
| `arms/A.ids`, `A2.ids` | M1: byte-identical 576-id streams (`cmp` clean) |
| `arms/A_det.err`, `A_det2.err` | M1 walls (49750.9 / 50370.5 s), TTFT 2460.54 s, spec-gate + lookup lines, final `Speed:` completion evidence (E18 M1 cache key) |
| `summary.json` | runner-generated band math: dRel(A), verdicts, note — regenerated 2026-10-30T16:42:30 by the E18 runner |

— written 2026-10-30 from the completed `summary.json` and re-verified against
every raw `arms/*` file (ppl, nats, walls, RSS, rc, banners, M1 `cmp`) before
commit, per §5.5.
