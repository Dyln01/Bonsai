# Cascade-Hot Streaming (CHS) — what is measurable, what ships, what does not

CHS proposes three explicit tiers (L0 pinned hot / L1 rolling window / L2 disk), a cheap
linear predictor of the next token's active set, and prefetch of only the **delta** between
consecutive tokens, overlapped with compute. Its projected effect on the field box is
`~60–90 s/token` (int4 + sorted + HOT_FRAC) → `~18–35 s/token` with the predictor.

This note records what this repo can already measure about that, what was implemented, and
what is withheld. The short version: **the delta exists, but it is ~0.6% of the bytes, and
the affordable predictor does not beat the baselines the engine already uses for free.**

Everything below is synthetic-container measured. Real-checkpoint re-runs are named as such.

## 1. What CHS adds over the machinery already here

| CHS element | Status before this note |
|---|---|
| L1 rolling window | Shipped (Phase 4b, `Q36_WIN_MB`), allocatable only on an int4 trunk |
| Candidate set (`HOT_FRAC`, GRIFFIN) | Shipped; the sorted container carries the prefix contract |
| Per-token dumps + ridge predictor | Shipped (`Q36_PROFILE_TOK` dumps, `scripts/exp_predictor_q36.py`) |
| Offline predictor decision rule | Pre-registered (`docs/PREDICTOR_QWEN36.md`): +5 pp over **both** static-A and GRIFFIN |
| L0 pinned tier, differential delta, async prefetch | **New in CHS.** Not previously present |

So CHS's genuinely new parts are the pinned tier and the delta/prefetch path. Both are
premised on a prediction that is affordable *and* accurate, which is measurable offline
from dumps the engine already writes.

## 2. The premise, measured

`tools/train_chs_predictor.py --delta-report` reads the dumps and reports how much the true
per-token top-K set actually moves (8 layers, 96 tokens, `K=64`, synthetic container):

```
mean delta 17.8 of 64 neurons/layer = 27.8% of the hot set
identical consecutive sets: 0/760 (0.0%)
```

So the premise is **not vacuous**: consecutive token sets are never identical and 27.8% of
the set turns over. Converted to traffic at field geometry (one neuron = gate/up/down rows
of width `D=5120` at int4 = 7 680 B; 64 layers):

    17.8 neurons/layer x 64 layers x 7 680 B  =  ~8.7 MB/token

For comparison, the per-token stream at int4 / `HOT_FRAC=0.3` is ~1.54 GB. **The delta is
0.57% of the bytes.** The design sketch's "read a few hundred MB of delta" is off by ~1–2
orders of magnitude, and that matters: a tier whose entire ceiling is 0.6% of traffic cannot
move 99 s/token to 18–35 s/token. At the field's ~10–11 MB/s that 8.7 MB is ~0.8 s/token.

Two further notes the same measurement forces:

* The churn is in the *index sets*; a set can swap 27.8% of its members and still carry the
  same energy. The correct quantity to chase is the energy-weighted delta, not the set delta.
* The dump writer selects the true set by **signed** per-token energy, while the shipped
  hot-profile accumulates `fabs(row[i])` (`engine/qwen36.c`). The engine's prefix and the
  dumps' "true set" are ranked by different quantities — worth reconciling before building
  anything that depends on one predicting the other.

## 3. The predictor, measured properly

The first run of this tool reported `static-A = 1.0000` and a predictor that *beat* it. Both
numbers were artifacts of the metric, now fixed in this tool **and** in
`scripts/exp_predictor_q36.py`:

1. `tmap[true_idx] = True` on a 2-D array marks the **union** of every token's true set. With
   enough tokens the union approaches the whole layer, so any prediction inside it scores as
   a hit and every arm reports ~1.0000. The mask must be per token.
2. The recorded energies are **signed** (~31% negative on random weights), so a signed sum is
   not a fraction: a set that hits only negative-energy neurons scores **> 1.0** and appears
   to beat a perfect baseline.

With a per-token magnitude metric (magnitude matches the engine's `fabs` aggregation), on the
same synthetic dumps (`K=64`, holdout eval, rank scan):

| rank | predictor | static-A | GRIFFIN-j | vs static | vs GRIFFIN | verdict | MB/layer |
|---|---|---|---|---|---|---|---|
| 16 | 0.6592 | 0.7288 | 0.7868 | −0.0696 | −0.1275 | FAIL | 0.05 |
| 64 | 0.7067 | 0.7288 | 0.7868 | −0.0221 | −0.0801 | FAIL | 0.20 |
| 256 | 0.7067 | 0.7288 | 0.7868 | −0.0221 | −0.0801 | FAIL | 0.79 |

Rank saturates at 64 (noise beyond it), and the predictor **loses to both baselines** — most
badly to GRIFFIN's per-prompt aggregate, which costs nothing. The pre-registered rule is not
met, so per the contract the runtime prefetch is **not unlocked**.

Note what these numbers now show that the broken metric hid: coverage is *not* saturated
(0.73 static, 0.79 GRIFFIN). There is real energy a better selection could capture — it is
the affordable *predictor* that fails, not the premise. A per-token oracle would be worth
~21 pp of true-set energy; nothing in this repo can select it at runtime for free.

## 4. What ships

* `tools/train_chs_predictor.py` — the offline half, against the **real** dump format
  (`<prefix>_L<l>.bin`, records of `D f32 | K u32 | K f32`; a bare-f32 read turns the index
  bytes into denormals and every index into 0). Ridge fit per layer, factorized to
  `A[D,r] @ B[r,I]` because the natural target `W[D,I]` is 356 MB/layer (22.8 GB over 64
  layers) and unloadable; reports the rank/reward curve and writes the artifact with
  `gate_cleared` so a runtime **cannot** silently prefetch from an unvalidated map.
  `--delta-report` is the premise check above.
* `tools/test_chs_predictor.py` — gates, all of which were violated at least once while
  building it: index round-trip through the structured dtype; the metric stays in [0,1] with
  signed energies; the churn report recovers a delta this test constructs exactly; a
  *constant* set cannot be beaten (and the tool must say so); and the counter-case — dumps
  whose set IS a linear function of `x` must **PASS**, so the gate is not trivially negative.
* Metric fixes in `scripts/exp_predictor_q36.py` (see §3). The Phase-4a ledger entry's
  numbers were produced with the union bug; its *verdict* (runtime not unlocked) is unchanged
  on the corrected metric, but the coverage figures in it should be re-derived on real dumps.

## 5. What does not ship, and the precondition

The engine-side path — L0 pinned set, delta computation, async prefetch — is **not**
implemented. Precondition to revisit: a real-container dump where the gate clears
(`gate_cleared=true` at a rank whose artifact fits the RAM budget). Even then the delta's
ceiling is the ~0.6% of bytes measured in §2, so the case for the *prefetch* machinery
remains weak; the case for **L0 pinning** does not depend on any predictor and is separable.

L0 pinning is the one part of CHS that needs no prediction: pin the highest-energy prefix in
RAM and never evict it. Two things it buys over the existing FIFO window are real — resident
from the first token, and immune to eviction by a scan — but both change *which* and *when*
bytes are resident, not *how many*. `project_q36.py --win-mb` already prices that byte count,
and the W-arm read-matrix (`scripts/check_window_readmatrix.py`, §14.1 of COMMANDS.md)
already measures the read-bytes cut, so L0 should be judged by running that matrix with a
larger resident budget before any engine work starts.

## 6. Run commands

```bash
# premise: how much does the true per-token set actually move?
python3 tools/train_chs_predictor.py --dumps ~/dumps/train --D 5120 --I 17408 --K 64 \
        --delta-report

# the pre-registered predictor gate, with the rank/reward curve
python3 tools/train_chs_predictor.py --dumps ~/dumps/train --dumps-eval ~/dumps/eval \
        --D 5120 --I 17408 --K 64 --rank-scan 16,32,64,128 --rank 64 \
        -o MODEL/chs_predictor.npz --json chs_g2.json

# the gates
python3 tools/test_chs_predictor.py
```

`--dumps-eval` is required for the pre-registered rule (disjoint corpus). Without it the tool
runs a same-corpus holdout and **says so** in the output and in the JSON, rather than
labelling a leaky split as the real gate.
