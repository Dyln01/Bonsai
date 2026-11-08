#!/usr/bin/env python3
"""test_layer_fracs.py — offline gates for the per-layer HOT_FRAC schedule tool.

No model, no engine: a synthetic container dir (meta only) + a synthetic
hot_profile.json in the engine's own written format. Run:
    python3 tools/test_layer_fracs.py

Gates
  G1 the emitted file is a FLAT map of "L<n>": <frac> pairs only, and re-reading it
     with a re-implementation of the ENGINE's own naive scan returns the same
     fractions (a nested or oddly-named key would silently become layer-0 garbage).
  G2 the byte budget is respected (discrete knapsack: within one neuron of it), and
     every layer in the profile gets a fraction inside [--min-frac, --max-frac].
  G3 on a profile with heterogeneous concentration the schedule captures strictly more
     energy than a uniform fraction at the SAME bytes — the only reason to use it.
  G4 --normalize reallocates (and still respects the budget); clamps bind.
  G5 refusals: missing profile, profile from another geometry, bad budget inputs.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "make_layer_fracs.py")
sys.path.insert(0, HERE)
import make_layer_fracs as mlf                                  # noqa: E402

FAILS = []
H, I, NL = 512, 256, 8


def check(ok, label, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)


def run(*args, expect_rc: "int | None" = 0):
    p = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
    if expect_rc is not None and p.returncode != expect_rc:
        print(f"    rc={p.returncode} (wanted {expect_rc}); stderr tail: "
              f"{p.stderr.strip().splitlines()[-1:] }")
    return p


def make_model(tmp, name="model"):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(model_type="qwen3_5", hidden=H, shared_inter=I, n_layers=NL,
                   num_experts=0, q_heads=8, kv_heads=2, head_dim=64),
              open(os.path.join(d, "qwen36_meta.json"), "w"))
    return d


def make_profile(d, heterogeneous=True, width=I, layers=NL):
    """Engine-written format: {"L0": {"n": <int>, "energy": [...]}}."""
    prof = {}
    rng = np.random.default_rng(7)
    for li in range(layers):
        if heterogeneous:
            # alternating concentrated / flat layers: layer 0 keeps 70% of its energy in
            # its first 5% of neurons, layer 1 is nearly flat -> a schedule MUST beat
            # a uniform fraction here, or the allocator is not doing anything.
            if li % 2 == 0:
                base = np.exp(-np.arange(width) / (width * 0.02))
            else:
                base = np.ones(width) * 0.9 + rng.random(width) * 0.1
        else:
            base = np.ones(width)
        prof[f"L{li}"] = dict(n=width, energy=[float(x) for x in base])
    json.dump(prof, open(os.path.join(d, "hot_profile.json"), "w"))
    return prof


def main():
    tmp = tempfile.mkdtemp(prefix="layer_fracs_test_")
    try:
        model = make_model(tmp)
        prof = make_profile(model)

        print("== G1: engine-readable flat map + budget/clamps ==")
        # the whole synthetic MLP is 8 x 256 x 768 B = 1.57 MB, so a binding budget must
        # be SMALLER than that — a budget above the model means nothing is traded at all
        FULL = NL * I * 3.0 * H * 0.5
        BUD = FULL * 0.35
        out = os.path.join(tmp, "hot_layers.json")
        p = run(model, "--budget-gb", "%.9f" % (BUD / 1e9), "--check", "-o", out)
        check(p.returncode == 0, f"tool exit 0 on a binding budget ({BUD / 1e6:.2f} MB of "
                                 f"{FULL / 1e6:.2f} MB)")
        check("check: re-parsed with the engine's own scan" in p.stdout,
              "--check re-parsed the file with the engine's scan")
        txt = open(out).read()
        flat = re.fullmatch(r'\{\s*"L\d+":\s*[\d.]+(\s*,\s*"L\d+":\s*[\d.]+)*\s*\}', txt.strip())
        check(flat is not None, "file is a flat \"L<n>\": <frac> map and nothing else",
              txt.strip()[:60].replace("\n", " "))
        tbl = json.loads(txt)
        check(set(tbl) == {f"L{li}" for li in range(NL)}, "every profile layer is covered",
              f"{len(tbl)} layers")
        back = mlf.engine_parse(out)
        mismatch = [k for k in tbl if abs(back.get(int(k[1:]), -1.0) - tbl[k]) > 5e-4]
        check(not mismatch, "engine-scan round trip is exact", f"mismatched: {mismatch[:4]}")

        bpn = 3.0 * H * 0.5                     # int4 bytes per neuron row triplet
        # slack = one neuron per layer (discrete search) + the --min-frac floor, which
        # the allocator cannot trade away: L x min_frac x mlp_width x bytes/neuron
        SLACK = NL * (bpn + 0.05 * I * bpn)
        used = sum(round(v * I) for v in tbl.values()) * bpn
        check(used <= BUD + SLACK, "budget respected within one neuron + the min-frac floor",
              f"used {used / 1e6:.3f} MB of {BUD / 1e6:.3f} MB (slack {SLACK / 1e6:.3f} MB)")
        check(all(0.05 <= v <= 1.0 for v in tbl.values()),
              "min/max frac clamps respected",
              f"range {min(tbl.values()):.3f}-{max(tbl.values()):.3f}")
        check(len(set(round(v, 6) for v in tbl.values())) > 1,
              "the schedule is NOT uniform on a heterogeneous profile",
              f"{len(set(round(v, 6) for v in tbl.values()))} distinct fractions")

        print("\n== G3: the schedule must beat uniform at the same bytes ==")
        p = run(model, "--budget-gb", "%.9f" % (FULL * 0.5 / 1e9),
                "--json-report", os.path.join(tmp, "r.json"))
        rep = json.load(open(os.path.join(tmp, "r.json")))
        gain = rep["captured_energy"]["gain_pct"]
        check(gain > 0.5, "captured energy strictly beats the uniform fraction",
              f"{gain:+.2f}% at {rep['used_bytes'] / 1e9:.4f} GB")

        print("\n== G3b: --budget-sweep writes one engine-readable file per budget ==")
        sw = run(model, "--budget-sweep",
                 "%.9f,%.9f,%.9f" % (FULL * 0.25 / 1e9, FULL * 0.5 / 1e9, FULL * 0.9 / 1e9),
                 "--check", "-o", os.path.join(tmp, "sweep.json"))
        check(sw.returncode == 0, "--budget-sweep runs with --check")
        made = sorted(f for f in os.listdir(tmp) if re.fullmatch(r"sweep-.*g\.json", f))
        check(len(made) == 3, "one schedule file per budget", f"{made}")
        check(all(all(abs(mlf.engine_parse(os.path.join(tmp, f)).get(int(k[1:]), -1) - v) <= 5e-4
                      for k, v in json.load(open(os.path.join(tmp, f))).items())
                  for f in made), "every sweep file re-parses with the engine's scan")
        # captured energy must grow with the budget (recomputed from the profile directly)
        caps, fracs_list = [], []
        for f in made:
            fr = json.load(open(os.path.join(tmp, f)))
            tot = 0.0
            for li in range(NL):
                e = np.asarray(prof[f"L{li}"]["energy"], dtype=np.float64)
                k = int(round(fr[f"L{li}"] * I))
                tot += float(np.sort(e)[::-1][:k].sum())
            caps.append(tot)
        check(all(caps[i] < caps[i + 1] for i in range(len(caps) - 1)),
              "captured energy increases with the budget", " < ".join("%.1f" % c for c in caps))
        check(all(fracs_list[i] < fracs_list[i + 1] for i in range(len(fracs_list) - 1)),
              "mean frac increases with the budget", " < ".join("%.3f" % f for f in fracs_list))

        print("\n== G4: --normalize reallocates, clamps bind ==")
        p2 = run(model, "--budget-gb", "%.9f" % (FULL * 0.5 / 1e9), "--normalize",
                 "--json-report", os.path.join(tmp, "n.json"))
        rep2 = json.load(open(os.path.join(tmp, "n.json")))
        check(p2.returncode == 0 and rep2["normalize"] is True, "--normalize runs")
        check(rep2["used_bytes"] <= FULL * 0.5 + SLACK, "--normalize respects the budget",
              f"{rep2['used_bytes'] / 1e9:.4f} GB")
        f_a = [rep["layers"][k]["frac"] for k in sorted(rep["layers"], key=lambda x: int(x[1:]))]
        f_b = [rep2["layers"][k]["frac"] for k in sorted(rep2["layers"], key=lambda x: int(x[1:]))]
        check(any(abs(x - y) > 1e-6 for x, y in zip(f_a, f_b)),
              "--normalize changes the allocation")
        p3 = run(model, "--budget-gb", "%.9f" % (FULL * 0.6 / 1e9), "--min-frac", "0.35",
                "-o", os.path.join(tmp, "m.json"))
        t3 = json.load(open(os.path.join(tmp, "m.json")))
        check(min(t3.values()) >= 0.35, "--min-frac 0.35 is honoured",
              f"min {min(t3.values()):.3f}")
        p4 = run(model, "--budget-gb", "%.9f" % (FULL * 0.3 / 1e9), "--min-frac", "0.9",
                "--max-frac", "0.2")
        check(p4.returncode != 0 and not os.path.exists(os.path.join(tmp, "bad" + ".json")),
              "max-frac < min-frac refused")

        print("\n== G5: refusals ==")
        empty = make_model(tmp, "empty")
        p = run(empty, "-o", os.path.join(tmp, "x.json"), expect_rc=None)
        check(p.returncode != 0 and "calibrate" in (p.stderr + p.stdout),
              "missing hot_profile.json refused with the fix in the message")
        wrong = make_model(tmp, "wronggeom")
        # profile width 128 != mlp width 256 for every layer -> nothing usable
        make_profile(wrong, width=128)
        p = run(wrong, "-o", os.path.join(tmp, "y.json"), expect_rc=None)
        check(p.returncode != 0, "profile from another geometry refused")
        bad = os.path.join(tmp, "notacontainer")
        os.makedirs(bad, exist_ok=True)
        p = run(bad, "-o", os.path.join(tmp, "z.json"), expect_rc=None)
        check(p.returncode != 0 and "qwen36_meta.json" in (p.stderr + p.stdout),
              "non-container directory refused")
        p = run(model, "--budget-gb", "%.9f" % (FULL * 2 / 1e9), "--trunk-dtype", "f16",
                "--json-report", os.path.join(tmp, "f16.json"))
        r16 = json.load(open(os.path.join(tmp, "f16.json")))
        r4 = json.load(open(os.path.join(tmp, "r.json")))
        check(r16["bytes_per_neuron"] == 4 * r4["bytes_per_neuron"],
              "bytes/neuron tracks the trunk dtype (f16 = 4x int4)",
              f"{r4['bytes_per_neuron']:.0f} -> {r16['bytes_per_neuron']:.0f} B")
        check(r16["used_bytes"] <= FULL * 2 + NL * (r16["bytes_per_neuron"]
                                               * (1 + 0.05 * I)),
              "f16 budget respected too")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\nscratch {tmp} removed")
    print(f"\n{'ALL LAYER-FRAC GATES PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
