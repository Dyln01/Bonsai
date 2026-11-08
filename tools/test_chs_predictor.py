#!/usr/bin/env python3
"""test_chs_predictor.py -- gates for train_chs_predictor.py.

Four properties, each of which was actually violated (or nearly) while writing the tool,
plus the counter-case that keeps the gate honest: a predictor that CAN win must PASS, or
the test would just be asserting "this tool always says no".

  G1 the loader round-trips indices through the record's structured dtype. Reading the
     record as bare f32 turns the u32 index bytes into denormals and every index to 0 --
     an all-zeros predictor that looks like a working one.
  G2 the coverage metric is a fraction in [0, 1] when energies are SIGNED. With a signed
     sum, a set that hits only negative-energy neurons scores > 1.0 and "beats" a perfect
     baseline (observed: +0.11 pp at rank 64 on the first run).
  G3 the churn report recovers a delta this test constructs exactly (rotate the set by 8 of
     64 -> delta 8.0), and reports 0 identical sets.
  G4 the verdict tracks reality: on dumps whose true set IS a linear function of x, the
     predictor clears the +5 pp rule (PASS, gate_cleared=true); on dumps whose set is a
     constant, it does not (FAIL, gate_cleared=false), and the artifact records which.

Run: TMPDIR=... python3 tools/test_chs_predictor.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "train_chs_predictor.py")
D, I, K, L, N = 64, 96, 8, 3, 240
BAD = 0


def check(ok, what, detail=""):
    global BAD
    print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
    if not ok:
        BAD += 1


def write_dumps(prefix, idx_of, en_of):
    dt = np.dtype([("x", np.float32, (D,)), ("idx", np.uint32, (K,)), ("en", np.float32, (K,))])
    for l in range(L):
        rec = np.zeros(N, dtype=dt)
        for t in range(N):
            rec["x"][t] = idx_of(t, l)[1]
            rec["idx"][t] = idx_of(t, l)[0]
            rec["en"][t] = en_of(t, l)
        rec.tofile(f"{prefix}_L{l}.bin")
    return prefix


def run(prefix, extra):
    return subprocess.run([sys.executable, TOOL, "--dumps", prefix, "--D", str(D), "--I", str(I),
                           "--K", str(K), "--layers", str(L)] + extra,
                          capture_output=True, text=True, cwd=HERE)


def main():
    tmp = tempfile.mkdtemp(prefix="chs_gate_")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(N, D)).astype(np.float32)

    print("== G1: the structured dtype round-trips indices ==")
    # a pure window that advances by exactly K each token: consecutive sets are DISJOINT,
    # so the churn delta is exactly K -- and the indices span a wide range of the layer.
    rot = write_dumps(os.path.join(tmp, "rot"),
                      lambda t, l: (np.mod(np.arange(K) + K * t + 5 * l, I).astype(np.uint32), X[t]),
                      lambda t, l: rng.normal(size=K).astype(np.float32))
    sys.path.insert(0, HERE)
    import train_chs_predictor as tr
    _, T, _ = tr.load(rot, D, K, L)
    check(np.array_equal(T[0][0], np.mod(np.arange(K) + 5 * 0, I)),
          "idx written == idx read back", f"{T[0][0][:5]}")
    check(T[0].dtype == np.uint32 and T[0].max() > K,
          "indices survive as real values across the dump (not all-zero denormals)",
          f"max={T[0].max()} over {T[0].shape[0]} tokens (a bare-f32 read gives 0)")

    print("\n== G2: coverage stays a fraction with signed energies ==")
    signed = rng.normal(size=(20, K)).astype(np.float32) * 3.0
    check((signed < 0).mean() > 0.2, "the test's energies really are signed",
          f"{(signed < 0).mean():.0%} negative")
    t_idx = np.tile(np.arange(K, dtype=np.uint32), (20, 1))
    perfect = tr.coverage(t_idx, t_idx, signed)
    empty = tr.coverage(np.tile(np.arange(K, 2 * K, dtype=np.uint32), (20, 1)), t_idx, signed)
    check(abs(perfect - 1.0) < 1e-6, "a perfect set scores exactly 1.0", f"{perfect:.4f}")
    check(empty <= 1.0, "a disjoint set cannot exceed 1.0", f"{empty:.4f}")

    print("\n== G3: the churn report recovers a constructed delta ==")
    r = run(rot, ["--delta-report"])
    check(r.returncode == 0, "--delta-report runs", r.stderr.strip()[:80])
    line = [l for l in r.stdout.splitlines() if "mean delta" in l]
    check(bool(line), "the report prints a delta line")
    got = float(line[0].split()[2]) if line else -1
    check(abs(got - float(K)) < 0.01,
          f"delta == {K}.0 neurons (window advances by K => disjoint, K={K})", f"got {got}")

    print("\n== G4: a predictor that CAN win must PASS ==")
    # x is a one-hot PHASE and the true set is that phase's window, so x -> set is linear and
    # learnable, while a static prefix (which sees every phase rotating past) cannot cover
    # any single token's energy. The gate must be able to clear +5 pp here.
    PH = 12
    def phase_x(t):
        v = np.zeros(D, dtype=np.float32); v[t % PH] = 1.0
        return v + 0.001 * rng.normal(size=D).astype(np.float32)
    def phase_set(t, l):
        base = ((t % PH) * K + 5 * l) % I
        return np.mod(np.arange(K) + base, I).astype(np.uint32)
    lin = write_dumps(os.path.join(tmp, "lin"),
                      lambda t, l: (phase_set(t, l), phase_x(t)),
                      lambda t, l: ((K - np.arange(K)) / K).astype(np.float32))
    r = run(lin, ["--rank", "16", "--rank-scan", "8,16", "-o", os.path.join(tmp, "p_lin.npz")])
    _, Tl, _ = tr.load(lin, D, K, L)
    check(len({tuple(r) for r in Tl[0]}) > 5, "the linear case really varies per token",
          f"{len({tuple(r) for r in Tl[0]})} distinct sets")
    z = np.load(os.path.join(tmp, "p_lin.npz"))
    check(bool(z["gate_cleared"][0]), "linear-set dumps -> gate_cleared=true",
          r.stdout.strip().splitlines()[-4] if r.returncode == 0 else r.stderr[:90])
    check(z["coverage_pred"][0] - z["coverage_static"][0] >= 0.05,
          "coverage beats static by >= 5 pp", f"{float(z['coverage_static'][0]):.4f} -> "
          f"{float(z['coverage_pred'][0]):.4f}")

    print("\n== G4b: a constant set cannot be beaten, and the tool must say so ==")
    const = write_dumps(os.path.join(tmp, "const"),
                        lambda t, l: (np.arange(K, dtype=np.uint32) + 3 * l, X[t]),
                        lambda t, l: rng.normal(size=K).astype(np.float32))
    r = run(const, ["--rank", "16", "-o", os.path.join(tmp, "p_const.npz")])
    z = np.load(os.path.join(tmp, "p_const.npz"))
    check(not bool(z["gate_cleared"][0]), "constant-set dumps -> gate_cleared=false")
    check(z["coverage_static"][0] >= 0.9999, "static-A is perfect on a constant set",
          f"{float(z['coverage_static'][0]):.4f}")
    check("NOT unlocked" in r.stdout and "gate_cleared=false" in r.stdout,
          "the verdict text says the runtime stays locked")
    js = os.path.join(tmp, "const.json")
    r = run(const, ["--rank", "16", "--json", js])
    rep = json.load(open(js))
    check(rep["gate_cleared"] is False and rep["written_rank"] == 16,
          "the JSON report carries the verdict and the written rank")

    print("\n== G5: a dump without the record's exact size is refused, not guessed ==")
    open(os.path.join(tmp, "bad_L0.bin"), "wb").write(b"\x00" * 37)
    r = run(os.path.join(tmp, "bad"), ["--delta-report"])
    check(r.returncode != 0, "a malformed dump exits non-zero", r.stderr.strip()[:90])

    print("\nCHS PREDICTOR GATES: " + ("PASS" if not BAD else f"{BAD} FAILED"))
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
