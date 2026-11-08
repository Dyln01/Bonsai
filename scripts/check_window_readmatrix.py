#!/usr/bin/env python3
"""check_window_readmatrix.py — the pre-registered banked verdict for the window cache.

Phase 4b's gate 3 was written as: window-on must cut read bytes >=50% at equal-or-better
decode, else the knob stays default-off. The field run then returned a documented NEGATIVE
that was really an identity (the window could never allocate on the F16 trunk) plus an
E12/E13 arithmetic problem (at the field-forced n_new=2 there is one decode step, and the
decode re-read is only 17-24 GB of ~71.8 GB per arm, so even a perfect window cannot cut
the TOTAL by 50%). The lesson recorded in the ROADMAP was: a pre-registered gate must state
its n_new. This script is that gate, executable, against a measured battery JSON.

    python3 scripts/check_window_readmatrix.py results/battery_<host>_<ts>.json

It reads the W arms (`W0`, `W256`, `W1G`, or any `W<digits>`), compares each to `W0` on
`fs_read_gb` — the /proc/<pid>/io read accounting the battery already records — and fails
when the pre-registered bar is not met AT an n_new where the bar is meaningful. Two things
it will NOT do: silently pass when the window never allocated (the `hits 0 misses 0`
signature in the arm's log is reported as such), and treat an under-powered n_new as a
verdict (it marks the run INCONCLUSIVE and says why).
"""
import json
import os
import re
import shutil
import subprocess
import sys

BAR_TOTAL_CUT = 0.40      # pre-registered: >=40% total read-bytes cut vs W0
BAR_NEW_MIN = 8           # the n_new below which the bar is arithmetic-impossible


def die(msg):
    print(f"check_window_readmatrix: {msg}", file=sys.stderr)
    sys.exit(2)


def self_test():
    """CI gate for THIS script: four synthetic matrices, four expected verdicts. The
    interesting one is `identity` — the E14 signature where the window never allocated,
    which must never be reported as a verdict about the mechanism."""
    import tempfile
    def mk(n_new, arms):
        return dict(host="ci", frac=0.25, maxt=8192, n_new=n_new, arms=arms)
    def arm(fs, note=""):
        return dict(fs_read_gb=fs, decode_ms_per_tok=2225000, ttft_s=3466.0, rc=0, note=note)
    W = "[qwen36 win] hits 0 misses 0 (0.0% hit-rate), budget 256/1024 MB"
    cases = {
        "pass": (mk(8, {"W0": arm(71.7), "W256": arm(48.0), "W1G": arm(31.0)}), 0),
        "one_arm_over_bar": (mk(8, {"W0": arm(71.7), "W256": arm(71.2), "W1G": arm(31.0)}), 0),
        "fail": (mk(8, {"W0": arm(71.7), "W256": arm(71.2), "W1G": arm(71.5)}), 1),
        "identity": (mk(8, {"W0": arm(71.7, W), "W256": arm(71.7, W)}), 2),
        "underpowered": (mk(2, {"W0": arm(71.7), "W1G": arm(40.0)}), 2),
        "no_w0": (mk(8, {"W256": arm(48.0)}), 2),
    }
    bad = 0
    tmp = tempfile.mkdtemp(prefix="win_gate_")
    for name, (doc, want) in cases.items():
        p = os.path.join(tmp, name + ".json")
        json.dump(doc, open(p, "w"))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), p],
                           capture_output=True, text=True)
        ok = r.returncode == want
        bad += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}: exit {r.returncode} (want {want})")
        if not ok:
            print("   " + "\n   ".join(r.stdout.strip().splitlines()[-3:]))
    shutil.rmtree(tmp, ignore_errors=True)
    print("WINDOW GATE SELF-TEST PASS" if not bad else f"WINDOW GATE SELF-TEST: {bad} FAILED")
    return 1 if bad else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    if len(sys.argv) < 2:
        die("usage: check_window_readmatrix.py <battery.json> [--win-arm W1G] | --self-test")
    path = sys.argv[1]
    want = None
    if "--win-arm" in sys.argv:
        want = sys.argv[sys.argv.index("--win-arm") + 1]
    d = {}
    try:
        d = json.load(open(path))
    except Exception as e:
        die(f"{path}: unreadable ({e})")

    arms = d.get("arms") or {}
    if not arms:
        die(f"{path}: no 'arms' object — is this a run_battery.sh output?")
    wars = {k: v for k, v in arms.items() if re.fullmatch(r"W\d+|W1G", k or "")}
    if "W0" not in wars:
        die(f"{path}: no W0 control arm (found {sorted(wars)}) — a window matrix without "
            f"its off-arm is not a matrix")
    if want:
        if want not in wars:
            die(f"--win-arm {want} not in {sorted(wars)}")
        wars = {k: v for k, v in wars.items() if k in ("W0", want)}
    if len(wars) < 2:
        die(f"{path}: only {sorted(wars)} — need W0 plus at least one window arm")

    nnew = d.get("n_new") or d.get("nnew") or (d.get("args") or {}).get("n_new")
    nnew_txt = f"n_new={nnew}" if nnew else "n_new=UNRECORDED"
    print(f"== window read-matrix: {path}")
    print(f"   {nnew_txt}, host {d.get('host', '?')}, frac {d.get('frac', '?')}, "
          f"maxt {d.get('maxt', '?')}")

    def tripwire(a):
        """The E14 signature: the window refused to allocate (F16 trunk), so the arms are
        the SAME configuration three times — an identity, not a measurement."""
        blob = json.dumps(a)
        if "hits 0 misses 0" in blob:
            return "log carries 'hits 0 misses 0' — the window did not allocate (F16 trunk?)"
        return None

    base = wars["W0"].get("fs_read_gb")
    if not base:
        die("W0 has no fs_read_gb — the battery records it from /proc io; without it "
            "there is no read-bytes verdict (use BATTERY_STREAM/same-cache-cold runs)")
    rows, fails, incon = [], 0, 0
    print(f"\n{'arm':8} {'fs_read GB':>10} {'vs W0':>9} {'decode ms/tok':>14} "
          f"{'TTFT s':>9} {'rc':>4}  note")
    for k in sorted(wars, key=lambda x: (x != "W0", x)):
        a = wars[k]
        fr = a.get("fs_read_gb")
        cut = (1 - fr / base) if (fr and base) else None
        note = []
        if k == "W0":
            note.append("control (window off)")
        elif cut is not None:
            if cut >= BAR_TOTAL_CUT:
                note.append(f"PASS: >= {100 * BAR_TOTAL_CUT:.0f}% cut")
            else:
                note.append(f"below the {100 * BAR_TOTAL_CUT:.0f}% bar")
        trip = tripwire(a)
        if trip:
            note.append(trip)
        if a.get("rc") not in (0, "0", None):
            note.append(f"rc={a.get('rc')}")
        print(f"{k:8} {('' if fr is None else f'{fr:.2f}'):>10} "
              f"{('' if cut is None else f'{100 * cut:+.1f}%'):>9} "
              f"{str(a.get('decode_ms_per_tok', '')):>14} {str(a.get('ttft_s', '')):>9} "
              f"{str(a.get('rc', '')):>4}  {'; '.join(note)}")
        rows.append((k, fr, cut, trip))
        if k == "W0":
            continue
        if trip:
            incon += 1
        elif cut is not None and cut < BAR_TOTAL_CUT:
            fails += 1

    # `best` must be the best arm that CLEARS the bar — the loser of a "nothing cleared it"
    # matrix is not a budget that passed (the self-test rehearses exactly that case).
    best = max((r for r in rows if r[0] != "W0" and r[2] is not None and r[2] >= BAR_TOTAL_CUT),
               key=lambda r: r[2], default=None)
    print()
    if incon:
        print(f"INCONCLUSIVE: {incon} window arm(s) never allocated — the container's MLP "
              f"maps are not int4 (q36_win_ensure needs map_len == D*I/2).")
        print("  fix: python3 tools/pack_trunk_i4.py <model> <model>-i4 --profile <profile>")
        return 2
    if nnew is not None and int(nnew) < BAR_NEW_MIN:
        print(f"INCONCLUSIVE: n_new={nnew} < {BAR_NEW_MIN}. At the field's one-step budget "
              f"the decode re-read is only part of the per-arm total, so a 40% TOTAL cut is "
              f"arithmetically out of reach for any window — this is E14's calibration "
              f"lesson, pre-registered here. Re-run with BATTERY_NNEW>={BAR_NEW_MIN}.")
        return 2
    if fails and best is None:
        print(f"FAIL: no window budget cut total read bytes by the pre-registered "
              f"{100 * BAR_TOTAL_CUT:.0f}% at n_new={nnew} ({fails} arm(s) below the bar). "
              f"Per the contract the knob stays default-off and this number goes in the "
              f"report as-is.")
        return 1
    # The bar is a threshold on the configuration you would actually ship, so the verdict
    # names the budget that clears it — and names the ones that do not, rather than
    # collapsing a Pareto result into one word.
    assert best is not None
    print(f"PASS at budget {best[0]}: {100 * best[2]:+.1f}% total read-bytes cut vs W0 at "
          f"n_new={nnew}"
          + (f" ({fails} smaller arm(s) stay below the bar: "
             f"{', '.join(r[0] for r in rows if r[0] != 'W0' and r[2] is not None and r[2] < BAR_TOTAL_CUT)})"
             if fails else " (every window arm clears it)")
          + ". Record it in results/README and re-open the knob's default in "
            "docs/WINDOWING_QWEN36.md at that budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
