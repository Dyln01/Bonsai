# q36-minipc v2 — everything integrated into the launcher

**Date:** 2026-10-16 · WSL `~/bin/q36-minipc` + `~/bin/hot_sort.py` (source: colibri-harness/)

## What "integrated" means now

One program drives the entire session's work. It auto-detects the container class from
`qwen36_meta.json` (`num_experts > 0` → MoE, else dense) and composes the right stack:

- **MoE profile** (exact routing, full capability): cap + `PIN=auto` + `PILOT=2` + `WIDE` +
  `KV8` + `CAP_RAISE=0` + thread cap + RAM auto-detect.
- **Dense profile** (full-size model, trunk on disk): `TRUNK_MAP=1` + `GRIFFIN=1` +
  `HOT_FRAC` (default 0.25; 1.0 = exact) + `TRUNK_WARM=1` + `KV8` — works on UNMODIFIED
  containers; optional `Q36_GRIFFIN=0` to use a hot-sorted prefix instead.

Commands: `serve` · `chat` · `bench` (honest phase-split battery) · `seed` (MoE prior) ·
**`calibrate`** (dense: HOT_PROFILE run over `Q36_CORPUS`) · **`sort`** (dense: hot_sort.py
permutes gate/up rows + down cols + row-scales by the measured profile into `<model>-sorted`)
· `doctor` (knob/model/level/ram report).

## Validated end-to-end (this session, no downloads)

| step | result |
|---|---|
| `bench` dense (auto profile) | class auto-detect ✓, TRUNK_MAP+GRIFFIN+FRAC+WARM composed ✓, 109 ms/tok, RSS 1.87 GB, `[warm] … 0.05 GB (frac 0.25)` |
| `calibrate` | profile written: 20 layers, 579 KB (`[hot] profile written`) |
| `sort` | 20 layers permuted → `student_1b_i4_lsorted` (spans re-packed, row-scales permuted) |
| `bench` sorted copy | **103 ms/tok (9.7 tok/s)**, RSS 1.87 GB — sorted prefix slightly faster than unsorted+griffin |
| `bench` MoE (sanity) | class moe ✓, hit 98.2%, TTFT 15.8 s — v1 behavior intact |
| `doctor` (no model / dense / moe) | all three report correctly |

## Bugs found and fixed during integration (all real, all worth remembering)

1. **`hot_sort.py` read_st span bug** — `data_offsets` is `[start, END]`, not `[start, length]`;
   reading `f.read(end)` over-read every tensor by its start offset → garbage sizes
   (2,905,232 "bytes" for a 2,211,840-byte tensor), reshape failures. Fix: `raw = f.read(off1-off0)`.
2. **int4 nibble convention** — byte k holds elements 2k (LOW) and 2k+1 (HIGH); unpacking must
   split ALL bytes into lo/hi nibbles (`flat & 0xF` / `flat >> 4`), not pairs of bytes.
3. **Packed tensors carry 1-D shapes** in these containers — derive I/H from `qwen36_meta.json`,
   never from the header shape.
4. **`help`/`doctor` must not require a model** — the arg guard now whitelists them.
5. Launcher body duplicated the class-detection block after the guard was introduced — removed.

## The complete one-command workflow (dense, full-size 27B class)

```bash
# day one, per model:
q36-minipc calibrate ~/q27_i4            # Q36_CORPUS=your_text.txt
q36-minipc sort      ~/q27_i4            # -> ~/q27_i4-sorted
# every day after:
q36-minipc serve    ~/q27_i4-sorted      # TRUNK_MAP+prefix/GRIFFIN+KV8, ~2 GB + hot set
q36-minipc bench    ~/q27_i4-sorted      # honest battery anytime
```

Or skip calibration entirely (GRIFFIN adapts per prompt): `q36-minipc serve ~/q27_i4`.
