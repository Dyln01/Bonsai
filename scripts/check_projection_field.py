#!/usr/bin/env python3
"""check_projection_field.py — the projection must reproduce the field. CI gate.

Phase 5 de-projected the README's "12-14 tok/s warm" row because it was arithmetic
for a container nobody ran. The durable fix is a projection that is (a) a function
of the host and the container, and (b) REGRESSION-GATED against the numbers the
field actually measured. This script is (b): it runs tools/project_q36.py at each
recorded field point and fails the build if the model drifts out of its band, then
asserts the two consequences the packer's whole point rests on:

  1. the F16 field container can never be warm on the 7.62 GB box at frac 0.25
     (the measured 2225 s/token regime), and
  2. an int4-packed trunk at the same frac moves the working set under the box's
     RAM, which is the only reason the warm regime is reachable at all.

No model, no weights, no disk: every recorded value is a committed artifact under
results/ (read here so the gate fails loudly if one is ever edited away).
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "tools", "project_q36.py")
BATTERY = os.path.join(ROOT, "results", "battery_llm-030_20261025-133007.json")
REALW = os.path.join(ROOT, "results", "realweights_20261026-1014")
SUMMARY = os.path.join(REALW, "summary.json")

FAILS = []


def check(ok, label, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)


def run(*args):
    p = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    print("== the recorded field numbers this gate is calibrated on ==")
    ok_artifacts = True
    for path in (BATTERY, SUMMARY):
        if not os.path.exists(path):
            check(False, f"field artifact present: {os.path.relpath(path, ROOT)}")
            ok_artifacts = False
    if ok_artifacts:
        txt = open(BATTERY).read()
        # the battery records decode as ms/token strings and fs_read per arm; both are
        # what the model's rate and window were derived from
        ms = [float(x) for x in re.findall(r"([\d.]+)\s*ms/tok", txt)]
        if ms:
            print(f"   {os.path.relpath(BATTERY, ROOT)}: {len(ms)} arm(s), ms/token "
                  f"{min(ms):.0f}-{max(ms):.0f} -> {min(ms) / 1000:.0f}-{max(ms) / 1000:.0f} s/token")
            check(min(ms) / 1000 > 2000, "battery decode is the ~2000+ s/token regime "
                                         "(the number this projection exists to predict)")
        fr = [float(x) for x in re.findall(r'"fs_read_gb":\s*([\d.]+)', txt)]
        if fr:
            print(f"   fs_read per arm: {min(fr):.1f}-{max(fr):.1f} GB")
        check(os.path.exists(SUMMARY), "realweights summary.json present")
        s = json.load(open(SUMMARY))
        keys = list(s) if isinstance(s, dict) else []
        print(f"   summary.json top-level keys: {keys[:8]}")
    print("\n== model vs field (tools/project_q36.py --field-check) ==")
    rc, out = run("--geometry", "--field-check")
    print("\n".join("   " + l for l in out.strip().splitlines()[-24:]))
    check(rc == 0, "project_q36 --field-check: every recorded point inside its band")

    print("\n== consequences the packer's point rests on ==")
    rc_f16, out_f16 = run("--geometry", "--trunk-dtype", "F16", "--small-dtype", "F16",
                          "--frac", "0.25", "--ram-gb", "7.62")
    rc_i4, out_i4 = run("--geometry", "--trunk-dtype", "I4", "--small-dtype", "I4",
                        "--attn-dtype", "I4", "--frac", "0.25", "--ram-gb", "7.62")

    def grab(txt, pat, cast=float):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else None

    f16_s = grab(out_f16, r"decode, cold[^(]*\(([\d.]+) s/token") or \
        grab(out_f16, r"decode, cold \(every token off disk\):\s*([\d.]+) s/token")
    i4_s = grab(out_i4, r"decode, cold \(every token off disk\):\s*([\d.]+) s/token")
    f16_wall = grab(out_f16, r"MLP prefix fits at frac <= ([\d.]+)")
    i4_wall = grab(out_i4, r"MLP prefix fits at frac <= ([\d.]+)")
    check(f16_s is not None and i4_s is not None, "both columns rendered",
          f"F16 {f16_s} s/token, int4 {i4_s} s/token")
    if f16_s and i4_s:
        check(i4_s * 3.0 < f16_s, "int4 trunk cuts predicted decode >=3x at equal frac",
              f"{f16_s:.0f} -> {i4_s:.0f} s/token ({f16_s / i4_s:.1f}x)")
    if f16_wall is not None:
        check(f16_wall < 0.25, "F16 trunk: frac 0.25 does NOT fit the 7.62 GB box "
                               "(the measured regime)",
              f"needs frac <= {f16_wall:.3f}")
    if i4_wall is not None:
        check(i4_wall >= 0.25, "int4 trunk: frac 0.25 DOES fit -> warm regime reachable",
              f"fits at frac <= {i4_wall:.3f}")
    check("NO dial" in out_f16 and "HOT_FRAC dial" in out_f16,
          "attribution names what the dial does and does not govern")
    print(f"\n{'PROJECTION GATE PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
