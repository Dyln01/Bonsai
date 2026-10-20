# MTP draft-cost battery (Phase 2b-4) — measured

**Status: shipped** (2026-10-20). Runner `tools/run_battery.sh` (six arms,
structured JSON out); results in `results/results_mtp_q36.json`.
Operating point: synth_qwen27, 15,960-token prompt, 64 new tokens, greedy,
4-core WSL2 Ryzen 7 7735HS (the standing battery operating point for every
report in this repo; the same runner gives literal mini-PC numbers with
`bash tools/run_battery.sh <model> <prompt>` on that box). All six arms rc=0.

## The KV-tier arms (context for the MTP arms)

| arm | config | decode ms/tok | TTFT s | RSS GB |
|-----|--------|---------------|--------|--------|
| A | f32 KV | 16.88 | 82.4 | 0.79 |
| B | KV8 | 17.51 | 76.4 | 0.66 |
| C | KV_TQ4c1, absorb OFF | 53.39 | 91.1 | 0.63 |
| D | KV_TQ4c1, absorb ON | **24.49** | 87.3 | 0.64 |

**The 3b-rot absorption measured end-to-end: TQ4 total decode 53.39 →
24.49 ms/tok (−54.1%)**, moving TQ4 from 3.05× KV8's decode cost to 1.40×
while keeping its RAM win (0.63–0.64 GB vs KV8's 0.66 and f32's 0.79). The
remaining 40% over KV8 is per-row codec work + the sub-f32 score precision,
not redundancy. This closes the measurement the `ABSORB_ROT` doc was
waiting on.

## The MTP arms

| arm | config | decode ms/tok | settle |
|-----|--------|---------------|--------|
| E | KV8 + MTP drafts=2, gate off | 16.22 | 1 tok/forward |
| F | KV8 + MTP drafts=2, **forced accept** | 19.45 | **3 tok/forward** |

### Per-draft cost and verify scaling (the gate's inputs)

Arm E's report line, verbatim:

```
[qwen36 MTP] run: acceptance 0.0% (0/123 drafts in 62 verifies)
            | gate T(1) 16.3 ms, rows 2:1.73 3:2.64, rho 0.50,
            | acceptance by position 0.00 0.60 0.50 0.50
```

- **T(1) = 16.3 ms** — the calibrated single-row forward.
- **T(2) = 1.73×T(1), T(3) = 2.64×T(1)** — verify cost by row count, i.e.
  batching 2–3 rows amortizes the fixed parts; the second row costs 0.73×
  a first, the third another 0.91×. The gate consumes exactly this curve
  when deciding whether to draft.
- 123 head drafts over 62 verifies = 2 drafts per verify as configured.

### Tokens per forward under forced acceptance

With `Q36_MTP_FORCE=accept` every verify settles 3 tokens per forward:
**64 tokens in ~21 forwards**. Per-forward cost at forced-accept is
19.45 × 3 ≈ 58.4 ms vs 3 × 17.51 ≈ 52.5 ms for three plain forwards —
**+11% worst-case machinery overhead** on this container, where acceptance
is *fake* (the oracle forces acceptance of drafts the random-weight head
did not genuinely predict; the head and the 3-row verify still run in
full).

### What 0.0% acceptance means (and does not mean)

On synthetic random weights the head's drafts are noise — acceptance is
meaningless by construction (the house rule since the MTP design doc; the
same reason the CI gates use forced-oracle id-identity instead of natural
acceptance). What this battery measures is the **cost side**: draft
overhead at 0% genuine acceptance is within the run-to-run noise band
(E 16.22 vs B 17.51, both ±5–9% on identical config), and forced-accept
bounds the machinery at +11%/token at 3 tokens/forward. Whether drafting
*pay*s depends on the real acceptance rate (typical MTP heads: 60–80%),
which only real weights can show — pre-registered as a Phase 5 arm.

## Runner usage

```bash
tools/run_battery.sh <model_dir> <prompt_file> [out_json]
# env: BATTERY_THREADS=4 BATTERY_NNEW=64 BATTERY_CAP=8
#      BATTERY_ARMS="A B C D E F"   (subset, e.g. "C D" for the absorb A/B)
```

Honest-decode note: `decode ms/tok` is the engine's `[timers]` decode-phase
TOTAL, never the `Speed:` line (which includes prefill). Note for Windows
checkouts: run the runner from an LF copy (or after the `.gitattributes`
LF fix lands in your tree) — CRLF copies die at `set -o pipefail`.
