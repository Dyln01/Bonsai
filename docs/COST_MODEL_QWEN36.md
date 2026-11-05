# COST_MODEL_QWEN36.md — the de-projected numbers, and the two tools that make them honest

**Status:** shipped 2026-11-02. Tools: `tools/pack_trunk_i4.py`, `tools/project_q36.py`.
Gates: `tools/test_pack_trunk_i4.py` (offline, 5 min), `scripts/check_projection_field.py`
(no model needed, seconds), both wired into `.github/workflows/engine-build.yml`.

Phase 5 de-projected the README's "12–14 tok/s warm" row: it was arithmetic for an int4
trunk whose frac-0.25 hot prefix fits the page cache, and the field container is F16
(54.7 GB trunk) on a 7.62 GB box. Deleting a wrong number is necessary and not sufficient.
This document is the replacement: the projection is now a **function of the host and the
container**, regression-gated against the field numbers it has to reproduce, and the only
structural lever on the decode floor is now a shipped offline tool.

---

## 1. Why the projection was wrong — both errors, with the arithmetic

Field decode at the daily point (frac 0.25, 1524 ctx, 4-core / 7.62 GB / rotating disk)
is **2,225–2,358 s/token**, and compute cores are <1% of a token: decode is pure weight
streaming. At the measured ~11 MB/s that is **~23 GB streamed per token**, matching the
measured 17–24 GB decode re-read exactly (`results/battery_llm-030_20261025-133007.json`).

Where those bytes go, from the container's own geometry (`tools/project_q36.py`):

| class | per-token bytes (F16) | share | has a dial? |
|---|---|---|---|
| trunk DeltaNet (`in_proj_qkv`, `in_proj_z`, `out_proj`) | 18.12 GB | 57.4% | **no** |
| trunk MLP (shared expert, frac 0.25 of 34.2 GB) | 8.56 GB | 27.1% | yes — `HOT_FRAC` |
| lm_head | 2.54 GB | 8.1% | no |
| trunk attention (q/k/v/o) | 2.35 GB | 7.4% | **no** |
| **TOTAL** | **31.57 GB** | | |

**`HOT_FRAC` governs 27% of the per-token bytes.** The other 73% (attention, DeltaNet,
lm_head) has no dial, no paging and no quantization on the mapped path — it streams in
full, every token. That was never visible in the README, which presented `HOT_FRAC` as
*the* mechanism.

Two independent multiplicative errors made the old projection four orders of magnitude
optimistic:

1. **dtype ×4** — the projection assumed an int4 trunk. The pinned converter quantizes
   routed experts only, and a dense 27B has none: the container named `q27_i4` carries an
   **F16** dense trunk. Its frac-0.25 MLP prefix alone is 8.56 GB — more than the box's
   entire 7.62 GB RAM.
2. **transport ×~1000** — the 9.7 tok/s student-analog measurement was RAM-resident
   (1.87 GB working set served at DDR bandwidth). The field box streams from a 10–12 MB/s
   rotating disk, and nothing can stay warm: the working set exceeds RAM.

Neither is a kernel problem, so no kernel work closes the gap. Only bytes/token and where
the bytes live.

## 2. The lever: `tools/pack_trunk_i4.py`

Packs the dense trunk to int4 **offline**, disk→disk, streaming (row blocks, ~2 MB of RAM,
so it runs on the field box itself). It needs **no engine change** — the mapped int4 branch
dispatches on `dtype == U8 && nbytes == I*O/2` alone.

The dialect (pinned in the tool's docstring, mirrored from `engine/qwen36.c`):
byte *k* holds element 2*k* in the low nibble and 2*k*+1 in the high nibble, two's
complement, row-major `[O][I]`; the scale sidecar `<name>.qs` is F32 with **one scale per
output row**; the quantizer is `amax/7` with `q = rint(clip(w·7/amax, −7, 7))` — exactly
`scripts/gen_student_i4.py`'s `quant_i4_row`, the layout every CI gate already reads.

It packs every dense GEMV tensor the engine loads through `load_tq(quantize=1)`:
`lm_head`, the MLP (dense `mlp.*` or MoE `mlp.shared_expert.*`), the MoE router, the
attention projections and the DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj`. Everything
else (norms, embed, `conv1d`, `b`/`a`, `dt_bias`, `A_log`, experts) is copied verbatim.

```bash
python3 tools/pack_trunk_i4.py ~/Models/q27_i4 --dry-run          # what it would rewrite
python3 tools/pack_trunk_i4.py ~/Models/q27_i4 ~/Models/q27_i4i \
        --profile ~/Models/q27_i4/hot_profile.json --verify --hash
```

**What it buys (measured on the synthetic field-dialect container, 51 tensors):**

| | before | after |
|---|---|---|
| dense trunk | 3.93× the packed size | — |
| MLP prefix at frac 0.25 | 8.56 GB | **2.14 GB** |
| 4-bit rel L2 vs the F16 source | — | 0.127–0.137 (random Gaussian weights: worst case) |

`--profile` writes the hotness-sorted prefix **in the same pass** (rows for `gate`/`up`,
columns for `down`, `argsort(-energy, stable)` — `hot_sort.py`'s exact rule). That kills
two birds: it is what makes `HOT_FRAC`'s contract hold (arm E's pending requirement), and
it replaces the separate `hot_sort` output that was disk-blocked at ~50 GB — the packed
sorted container is ~14 GB, which fits the box's ~42 GB free.

Three things were blocked on the F16 trunk; one tool unblocks all three:

* the **per-layer window cache** — `q36_win_ensure` requires `map_len == D*I/2` maps, which
  is precisely why the E14 field arms logged `hits 0 misses 0` and Phase 4b closed as a
  documented negative;
* **arm E** (sorted container) — disk-blocked until now;
* honest **per-class byte accounting**, via `--dry-run` and `project_q36.py`.

### Verification (no engine needed — `tools/test_pack_trunk_i4.py`)

| gate | what it proves |
|---|---|
| G1 | the packed bytes, dequantized in *this file's own* convention, reproduce the `--emit-dequant` oracle container **exactly**, and are within the 4-bit bound of the F16 source per class |
| G2 | `pack(--profile)` ∘ `hot_sort` == `hot_sort` ∘ `pack(--profile)`, **byte-for-byte across 102 tensors** — nibble order, scale placement and the hotness permutation all agree with the shipped sorter |
| G3 | every target is `U8` with `nbytes == I*O/2` and an `F32` `.qs` of `numel == O` (the engine's dispatch predicate) |
| G4 | an already-int4 container is **refused** (never a silent double-pack); a profile from another geometry is a **refusal**, not a silently-unsorted container; a partial profile warns which layers would serve unsorted |

The `--emit-dequant` oracle is the point of G1/G2: the engine's mapped-F16 read of exactly
the values the mapped-int4 read reconstructs isolates layout bugs from all other numerics.

## 3. The replacement projection: `tools/project_q36.py`

```
bytes/token   analytic, from the container's own shard headers (per class, dtype-aware)
seconds/token (bytes − resident working set) / rate(host)
TTFT          prefill bytes / rate(host) + 105 s   (E13's measured constant)
warm case     the part of the working set that fits RAM is served at RAM rate; the rest
              still streams off the disk
```

All host parameters are **field anchors** with provenance, printed on every run:

| parameter | value | anchor |
|---|---|---|
| rate | 11.2 MB/s | E12/E13/E14 effective trunk-stream rate, 10.3–11.6 MB/s |
| resident working set | 6.9 GB, capped by RAM-free-after-residents | decode re-read 17–24 GB vs 31.6 GB analytic |
| prefill constant | 105 s | E13 per-arm constant |

**Field fit** (`--field-check`, and the CI gate `scripts/check_projection_field.py`):

| point | recorded | model | residual | band |
|---|---|---|---|---|
| decode, frac 0.25 @1524 ctx | 2225 s/token | 2384 s/token | +7.1% | ±15% |
| TTFT, frac 0.25 @1524 ctx | 3466 s | 2924 s | −15.7% | ±20% |
| TTFT, frac 1.0 @512 ctx | 2460.5 s | 5215 s | +112% | ±130% |

The third row is a **reported outlier, not a hidden one**: that run's per-byte prefill rate
is ~2× the battery's (page cache warmed by the preceding tf arm in the same session, plus
the ±25% HDD luck the report already records between arms). The model is calibrated on the
battery; the wide band exists so the disagreement stays on the record instead of being
tuned away.

`--probe <file>` replaces both rates with this host's own measured sequential read rate
(two passes: cold and page-cached), so the same tool answers the question on a different
box without pretending the anchors transfer.

## 4. What the projection says now

Same container, same box, same frac 0.25:

| | F16 trunk (measured) | int4 trunk (projected) |
|---|---|---|
| per-token bytes | 31.57 GB | 7.89 GB |
| off-disk per token | 24.67 GB | 3.02 GB |
| decode | 2384 s/token | **99 s/token** |
| MLP prefix | 8.56 GB → needs frac ≤ 0.142 to fit 7.62 GB | **2.14 GB → fits at frac ≤ 0.792** |

The second column is the whole point: at int4/frac 0.25 the working set finally fits the
box's RAM, which is the *only* reason a warm regime exists on this hardware at all. The
model says 99 s/token (still disk-bound, ~1.7 min/token); the warm column (hotset served
at RAM rate) is where tok/s lives, and reaching it needs the sorted prefix (`--profile`)
plus a frac chosen from the fit, not a constant.

**Honest headline, replacing the deleted one:** on an 8 GB / rotating-disk mini-PC the
realistic destinations are (a) *it runs at all* — full-size dense 27B, full compute,
ppl 6.74, rc=0, ≤7.01 GB RSS; (b) ~1–4 min/token with an int4 trunk and no sorted prefix;
(c) single-digit tok/s only when the working set is page-cache resident — which needs the
int4 trunk **and** a frac at or below the fit **and** (preferably) an NVMe host. Anything
quoting a tok/s figure without naming bytes/token, the medium and the resident set is the
old mistake in a new place.

## 5. What this does not do

* The byte accounting is structural; the ~11 MB/s rate is this host's. On NVMe the model
  will be wildly conservative — run `--probe` there.
* int4 is a real precision change on the trunk (rel L2 ~0.13 on random Gaussian weights,
  worst case). Per-component selection (`--components`) exists because the measured cost
  per byte differs by component (attention and DeltaNet input projections cost the most
  perplexity, lm_head the least — `COLI_DENSE_INT4`'s rationale); the trunk-level quality
  measurement is a Phase-5-class arm (wikitext-2 ppl at frac 1.0, same 512/64 window as
  arm A) and is **not** claimed here.
* G2/T4 prove the *machinery* agrees with the shipped sorter and with the engine's read
  predicate. They do not prove perplexity; the packaged sorted container still needs the
  arm-E run when disk allows.
