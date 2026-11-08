#!/usr/bin/env python3
"""make_layer_fracs.py — per-layer HOT_FRAC schedule from a calibration profile.

WHY. `HOT_FRAC` is one global scalar, but layers are not equally compressible: the
calibration pass (`q36-minipc calibrate` -> hot_profile.json) already measures each
MLP neuron's activation energy, and the energy curves differ layer to layer. Spending
the same bytes on every layer wastes them where a layer's energy is flat and starves
layers where it is concentrated. The bytes are also exactly comparable across layers
(per-neuron cost = 3 x hidden x bytes/element, identical for every layer of one model),
so "which neuron is worth its bytes" is a global knapsack over per-layer marginal
energies.

WHAT IT PRODUCES, and why it looks like this: the engine reads HOT_LAYERS with a naive
scan (`q36_layer_frac` in qwen36.c: strstr("\\\"L") -> atoi -> strchr(':') -> atof), so the
file must be a FLAT map of "L<n>": <frac> pairs and nothing else — a nested
{"layers": {"L0": ...}} happens to work only because lowercase "l" is not matched, and
any other key starting with "L" (e.g. "Layers") would be read as layer 0 with a garbage
fraction. We emit exactly what the parser can read, and validate the round trip against
a re-implementation of that parser before writing.

ALLOCATION. Maximise captured energy subject to a resident-byte budget:
    per-neuron bytes   = 3 * hidden * esz (0.5 for a packed int4 trunk, 2 for F16)
    K_l(lam)           = #{neurons in layer l with marginal energy >= lam}
    total(K(lam))      = sum_l K_l(lam)  ->  monotone  ->  binary-search lam for the budget
This is the standard Lagrangian relaxation of the knapsack: it is optimal up to one
neuron's granularity, and it is what makes the schedule beat a uniform fraction at the
same bytes. `--normalize` switches the objective to each layer's *relative* energy
share (treats layers as equally important) — the report prints both captured-energy
shares so the choice is measured rather than asserted.

USAGE
  python3 tools/make_layer_fracs.py <model_dir> --budget-gb 5.5 -o hot_layers.json
     --profile <path>   hot_profile.json (default <model_dir>/hot_profile.json)
     --budget-gb F      resident MLP bytes to spend (default: fit --ram-gb)
     --ram-gb F         derive the budget from RAM minus fixed residents (default 7.62)
     --trunk-dtype      int4 (default) | f16 — decides bytes/neuron
     --min-frac/--max-frac  clamps (default 0.05 / 1.0)
     --normalize        allocate by per-layer relative energy instead of absolute
     --json-report <p>  full report (per-layer K, captured energy, bytes)
     --check            also verify the file re-reads to the fractions we wrote

Output: the HOT_LAYERS JSON + a table showing captured energy vs a uniform fraction
at the same total bytes (the number that justifies using it at all).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ESZ = {"int4": 0.5, "i4": 0.5, "f16": 2.0, "f32": 4.0}
FIXED_RESIDENT_GB = 2.75          # embed + KV + norms, the field container's measured share


def die(msg):
    print(f"make_layer_fracs: error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_profile(path, width):
    """hot_profile.json -> {layer_index: energy array} (float64, length == width)."""
    if not os.path.exists(path):
        die(f"{path}: not found — run `q36-minipc calibrate <model>` first "
            f"(it writes hot_profile.json via HOT_PROFILE=1)")
    prof = json.load(open(path))
    out, skipped = {}, 0
    for k, v in prof.items():
        if not (isinstance(k, str) and k.startswith("L") and k[1:].isdigit()):
            continue
        if not isinstance(v, dict) or "energy" not in v:
            continue
        e = np.asarray(v["energy"], dtype=np.float64)
        if e.size != width:
            skipped += 1
            continue
        out[int(k[1:])] = e
    if not out:
        die(f"{path}: no usable \"L<n>\": {{energy: [...]}} entries at width {width} "
            f"({skipped} skipped on width mismatch) — is this a profile from this model?")
    if skipped:
        print(f"note: {skipped} layer(s) skipped (profile width != mlp width {width})",
              file=sys.stderr)
    return out


def geo_of(model_dir):
    mp = os.path.join(model_dir, "qwen36_meta.json")
    if not os.path.exists(mp):
        die(f"{model_dir}: no qwen36_meta.json — point this at a colibri container dir")
    m = json.load(open(mp))
    hidden = int(m.get("hidden") or 0)
    inter = int(m.get("shared_inter") or m.get("moe_inter") or 0)
    layers = int(m.get("n_layers") or 0)
    if not (hidden and inter and layers):
        die(f"{mp}: needs hidden / shared_inter (or moe_inter) / n_layers")
    return hidden, inter, layers


def engine_parse(path):
    """Re-implementation of q36_layer_frac's scan, so --check proves the ENGINE can read
    what we wrote (a nested or oddly-named key silently becomes layer 0 garbage)."""
    buf = open(path, "r").read()
    tbl, s = {}, 0
    while True:
        i = buf.find('"L', s)
        if i < 0:
            break
        j = i + 2
        digits = ""
        while j < len(buf) and buf[j].isdigit():
            digits += buf[j]
            j += 1
        c = buf.find(":", i)
        if c < 0:
            break
        ln = int(digits) if digits else 0
        k = c + 1
        while k < len(buf) and buf[k] in " \t\r\n":
            k += 1
        m = k
        while m < len(buf) and buf[m] in "0123456789.eE+-":
            m += 1
        try:
            v = float(buf[k:m])
        except ValueError:
            v = 0.0
        if 0 <= ln < 512 and 0.0 < v <= 1.0:
            tbl[ln] = v
        s = c + 1
    return tbl


def waterfill(energies, budget_bytes, bytes_per_neuron, min_frac, max_frac, normalize):
    """Bidirectional search on the marginal-energy threshold lam -> {layer: K}.
    Returns (K_by_layer, lam). Monotone in lam: higher lam -> fewer neurons."""
    L = sorted(energies)
    per_layer = {li: np.sort(energies[li])[::-1] for li in L}      # desc marginal values
    caps = {li: (int(len(per_layer[li])), max(1, int(round(len(per_layer[li]) * min_frac))),
                 int(round(len(per_layer[li]) * max_frac))) for li in L}

    def total_neurons(lam):
        n = 0
        for li in L:
            arr = per_layer[li]
            k = int(np.searchsorted(-arr, -lam, side="right")) if lam > 0 else arr.size
            n += min(max(k, caps[li][1]), caps[li][2])
        return n

    allv = np.concatenate([per_layer[li] for li in L])
    if normalize:
        allv = np.concatenate([per_layer[li] / max(per_layer[li][0], 1e-30) for li in L])
    hi = float(allv.max()) if allv.size else 1.0
    lo = 0.0
    target = budget_bytes / bytes_per_neuron
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if total_neurons(mid) > target:
            lo = mid
        else:
            hi = mid
    lam = hi
    K = {}
    for li in L:
        arr = per_layer[li]
        v = arr / max(arr[0], 1e-30) if normalize else arr
        k = int(np.searchsorted(-v, -lam, side="right")) if lam > 0 else arr.size
        K[li] = min(max(k, caps[li][1]), caps[li][2])
    return K, lam


def sweep(a, hidden, inter, energies, bytes_per_neuron, _geo):
    """One schedule per budget + the Pareto row for each: how much captured energy the
    bytes actually buy. This is the measurement plan's first half; the second half (ppl
    per schedule) is the Phase-5-class arm set, run against these files."""
    try:
        budgets = [float(x) for x in a.budget_sweep.split(",") if x.strip()]
    except ValueError:
        die(f"--budget-sweep wants comma-separated GB values, got '{a.budget_sweep}'")
    if not budgets:
        die("--budget-sweep: no values")
    stem = (a.out or os.path.join(a.model_dir, "hot_layers.json"))
    stem = re.sub(r"\.json$", "", stem)
    print(f"\n== budget sweep: {len(budgets)} schedule(s), "
          f"{bytes_per_neuron / 1024:.1f} KiB/neuron at {a.trunk_dtype} ==")
    print(f"{'budget GB':>9} {'used GB':>8} {'mean frac':>9} {'min':>6} {'max':>6} "
          f"{'captured':>12} {'vs uniform':>11}  file")
    rows = []
    for gb in budgets:
        K, lam = waterfill(energies, gb * 1e9, bytes_per_neuron, a.min_frac, a.max_frac, a.normalize)
        used = sum(K.values()) * bytes_per_neuron
        fracs = {li: K[li] / inter for li in sorted(K)}

        def cap(Kd):
            return sum(float(np.sort(arr)[::-1][:Kd.get(li, 0)].sum())
                       for li, arr in energies.items())
        c_sched = cap(K)
        uk = max(1, int(used / bytes_per_neuron / max(len(K), 1)))
        c_unif = cap({li: min(inter, uk) for li in K})
        path = f"{stem}-{gb:g}g.json"
        with open(path, "w") as f:
            f.write(json.dumps({f"L{li}": round(fracs[li], 6) for li in sorted(fracs)}, indent=1) + "\n")
        if a.check:
            back = engine_parse(path)
            bad = [li for li in fracs if abs(back.get(li, -1) - fracs[li]) > 5e-4]
            if bad:
                die(f"{path}: the engine's parser reads different values for {bad[:5]}")
        rows.append((gb, used, fracs, c_sched, c_unif, path))
        tag = ""
        if used > gb * 1e9 * 1.02:
            tag = "  <- floor-bound (--min-frac cannot be traded away)"
        print(f"{gb:9.2f} {used / 1e9:8.3f} {np.mean(list(fracs.values())):9.3f} "
              f"{min(fracs.values()):6.3f} {max(fracs.values()):6.3f} {c_sched:12.6g} "
              f"{100 * (c_sched / max(c_unif, 1e-30) - 1):+10.2f}%  {path}{tag}")
    if any(r[1] > r[0] * 1e9 * 1.02 for r in rows):
        print("\nnote: the smallest budgets are floor-bound — every layer is already at "
              "--min-frac, so lowering the budget further buys nothing.")
    print("\nEach file is a complete HOT_LAYERS map: serve it with Q36_HOT_LAYERS=<file>")
    print("(or copy it to <model>/hot_layers.json) and run the Phase-5-class ppl arms against")
    print("each — that pair of columns is the Pareto the schedule is chosen from.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--profile")
    ap.add_argument("--budget-gb", type=float)
    ap.add_argument("--ram-gb", type=float, default=7.62)
    ap.add_argument("--trunk-dtype", default="int4", choices=["int4", "f16", "f32"])
    ap.add_argument("--min-frac", type=float, default=0.05)
    ap.add_argument("--max-frac", type=float, default=1.0)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--json-report")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--budget-sweep", metavar="GB,GB,...",
                    help="write ONE schedule per budget (comma list) beside -o, named "
                         "<stem>-<gb>g.json, and print the bytes/energy Pareto row for each "
                         "— the resident-budget curve the quality arms are then run against")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    hidden, inter, n_layers = geo_of(a.model_dir)
    prof_path = a.profile or os.path.join(a.model_dir, "hot_profile.json")
    energies = load_profile(prof_path, inter)
    esz = ESZ[a.trunk_dtype]
    bytes_per_neuron = 3.0 * hidden * esz
    budget = (a.budget_gb * 1e9 if a.budget_gb else
              max(0.5e9, (a.ram_gb - FIXED_RESIDENT_GB) * 1e9))
    print(f"model    {a.model_dir}")
    print(f"geometry hidden {hidden} x mlp {inter} x {n_layers} layers"
          f"{'' if len(energies) == n_layers else f' (profile covers {len(energies)})'}")
    print(f"trunk    {a.trunk_dtype}: {bytes_per_neuron / 1024:.1f} KiB per neuron "
          f"(gate+up+down x hidden x {esz} B)")
    print(f"budget   {budget / 1e9:.2f} GB resident MLP bytes "
          f"{'(--budget-gb)' if a.budget_gb else f'(ram {a.ram_gb} - {FIXED_RESIDENT_GB} fixed)'}")

    if a.max_frac < a.min_frac:
        die("--max-frac < --min-frac")
    if a.budget_sweep:
        return sweep(a, hidden, inter, energies, bytes_per_neuron, geo_of(a.model_dir))
    full_bytes = len(energies) * inter * bytes_per_neuron
    if budget >= full_bytes:
        # Nothing to trade: every neuron fits. Say so instead of emitting a schedule
        # produced by the search's residual rounding.
        print(f"\nnote: the budget ({budget / 1e9:.2f} GB) covers the ENTIRE MLP "
              f"({full_bytes / 1e9:.2f} GB at {a.trunk_dtype}) — every layer gets "
              f"frac {a.max_frac:g}, so the schedule is not buying anything. Lower "
              f"--budget-gb (or --ram-gb) to make it bind.")
        K = {li: min(inter, int(round(inter * min(1.0, a.max_frac)))) for li in energies}
        lam = 0.0
    else:
        K, lam = waterfill(energies, budget, bytes_per_neuron, a.min_frac, a.max_frac, a.normalize)
    used = sum(K.values()) * bytes_per_neuron
    fracs = {li: K[li] / inter for li in sorted(K)}
    if used > budget * 1.02:
        floor_bytes = len(K) * max(1, int(round(inter * a.min_frac))) * bytes_per_neuron
        print(f"note: the schedule is {100 * (used / budget - 1):.1f}% over budget — the "
              f"per-layer --min-frac floor ({a.min_frac:g} x {len(K)} layers = "
              f"{floor_bytes / 1e6:.1f} MB) cannot be traded away. Raise --budget-gb or "
              f"lower --min-frac.", file=sys.stderr)

    # captured energy: absolute, and versus a uniform fraction at the same bytes
    def capture(Kd, norm=False):
        tot = 0.0
        for li, arr in energies.items():
            s = np.sort(arr)[::-1]
            tot += float(s[:Kd.get(li, 0)].sum())
        return tot
    cap_sched = capture(K)
    unif_k = {li: min(inter, max(1, int(used / bytes_per_neuron / len(K)))) for li in K}
    cap_unif = capture(unif_k)
    frac_unif = unif_k[min(unif_k)] / inter if unif_k else 0.0
    print(f"\nschedule: mean frac {np.mean(list(fracs.values())):.3f} "
          f"(min {min(fracs.values()):.3f} / max {max(fracs.values()):.3f}), "
          f"{used / 1e9:.2f} GB, lambda {lam:.3g}")
    print(f"captured energy: schedule {cap_sched:.6g} vs uniform-frac {cap_unif:.6g} "
          f"at the same bytes -> {100 * (cap_sched / max(cap_unif, 1e-30) - 1):+.2f}%")

    out = {f"L{li}": round(fracs[li], 6) for li in sorted(fracs)}
    text = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {a.out} ({len(out)} layers, flat \"L<n>\": <frac> pairs)")
        if a.check:
            back = engine_parse(a.out)
            bad = [li for li in fracs
                   if abs(back.get(li, -1) - fracs[li]) > 5e-4]
            if bad:
                die(f"the engine's parser reads different values for layers {bad[:6]} — "
                    f"the file must contain only \"L<n>\": <frac> pairs")
            print(f"check: re-parsed with the engine's own scan — all {len(back)} layers match")
    else:
        print("\n" + text)
    if a.json_report:
        json.dump(dict(model_dir=os.path.abspath(a.model_dir), profile=os.path.abspath(prof_path),
                       trunk_dtype=a.trunk_dtype, bytes_per_neuron=bytes_per_neuron,
                       budget_bytes=budget, used_bytes=used, lambda_=lam,
                       normalize=a.normalize, hidden=hidden, mlp_inter=inter,
                       layers={f"L{li}": dict(frac=fracs[li], K=K[li],
                                              energy=float(np.sort(energies[li])[::-1][:K[li]].sum()))
                               for li in sorted(fracs)},
                       captured_energy=dict(schedule=cap_sched, uniform=cap_unif,
                                            gain_pct=100 * (cap_sched / max(cap_unif, 1e-30) - 1),
                                            uniform_frac=frac_unif)),
                  open(a.json_report, "w"), indent=1)
        print(f"report {a.json_report}")


if __name__ == "__main__":
    main()
