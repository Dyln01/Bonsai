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


def floor_gate():
    """--floor must be arithmetic, not narrative: check it against closed-form cases."""
    import tempfile
    bad = 0

    def run(args):
        r = subprocess.run([sys.executable, TOOL] + args, capture_output=True, text=True)
        return r

    def check(ok, what, detail=""):
        nonlocal bad
        print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
        if not ok:
            bad += 1

    base = ["--geometry", "--trunk-dtype", "U8", "--frac", "0.25"]

    def streamed(out):
        m = re.search(r"now: streamed ([\d.]+) GB", out)
        return float(m.group(1)) if m else None

    # (a) a box big enough for everything must report 0 bytes streamed (compute-bound),
    #     and a 7.62 GB one must not.
    big = run(base + ["--small-dtype", "U8", "--ram-gb", "400", "--floor"])
    small = run(base + ["--ram-gb", "7.62", "--floor"])
    s_big, s_small = streamed(big.stdout), streamed(small.stdout)
    check(big.returncode == 0 and s_big == 0.0,
          "a box that fits the working set -> 0 GB streamed (compute-bound)", f"{s_big} GB")
    check(s_small is not None and s_small > 0,
          "a 7.62 GB box does stream at frac 0.25 with a wide lm_head", f"{s_small} GB")
    # (b) the JSON must satisfy streamed == max(0, total - room)
    out = os.path.join(tempfile.mkdtemp(prefix="floor_"), "f.json")
    run(base + ["--small-dtype", "F16", "--ram-gb", "7.62", "--floor", "--json", out])
    d = json.load(open(out))
    check(abs(d["streamed_now"] - max(0.0, d["total_bytes"] - d["cache_room"])) < 1.0,
          "streamed == max(0, total - cache_room)",
          f"{d['streamed_now'] / 1e9:.3f} vs "
          f"{max(0.0, d['total_bytes'] - d['cache_room']) / 1e9:.3f} GB")
    # (c) the dial is a minority and the no-dial classes dominate -- the reason frac
    #     tuning cannot reach the floor
    mlp_share = d["mlp_bytes"] / d["total_bytes"]
    check(0.0 < mlp_share < 0.5, "the MLP dial is a minority of the bytes",
          f"{100 * mlp_share:.0f}% dial-able")
    check(d["no_dial_bytes"] > d["mlp_bytes"],
          "the no-dial classes dominate (which is why frac tuning is out of road)")
    # (d) lm_head F16 -> int4 removes its byte difference. NOTE: --small-dtype drives BOTH
    #     lm_head and embed (the first version of this gate forgot that and expected half
    #     the observed delta), so hold the embed fixed by streaming it in both arms.
    out2 = out.replace(".json", "2.json")
    a_f16 = base + ["--small-dtype", "F16", "--ram-gb", "7.62", "--embed-streamed"]
    a_u8 = base + ["--small-dtype", "U8", "--ram-gb", "7.62", "--embed-streamed"]
    run(a_f16 + ["--floor", "--json", out])
    run(a_u8 + ["--floor", "--json", out2])
    d, d2 = json.load(open(out)), json.load(open(out2))
    cut = 1.907e9
    dr = d["streamed_now"] - d2["streamed_now"]
    check(abs(dr - cut) / cut < 0.02,
          "lm_head F16 -> int4 removes ~1.91 GB/token with the embed held fixed",
          f"{dr / 1e9:.3f} GB")
    check(abs((d["cache_room"] - d2["cache_room"])) < 1.0,
          "...and the arms really do hold the same cache_room (embed streamed in both)",
          f"{d['cache_room'] / 1e9:.2f} vs {d2['cache_room'] / 1e9:.2f} GB")
    # (e) freeing the embed must strictly increase the room and never decrease it
    out3 = out.replace(".json", "3.json")
    out4 = out.replace(".json", "4.json")
    run(base + ["--small-dtype", "F16", "--ram-gb", "7.62", "--floor", "--embed-streamed",
                "--json", out3])
    run(base + ["--small-dtype", "F16", "--ram-gb", "7.62", "--floor", "--json", out4])
    d3, d_base = json.load(open(out3)), json.load(open(out4))
    check(d3["cache_room"] > d_base["cache_room"],
          "streaming the embed row grows cache_room",
          f"{d_base['cache_room'] / 1e9:.2f} -> {d3['cache_room'] / 1e9:.2f} GB")
    check(d3["streamed_now"] <= d_base["streamed_now"],
          "and never increases what streams",
          f"{d_base['streamed_now'] / 1e9:.3f} -> {d3['streamed_now'] / 1e9:.3f} GB")

    print("FLOOR GATE: " + ("PASS" if not bad else f"{bad} FAILED"))
    return 1 if bad else 0


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

    print("\n== the stack ladder (each row is model output, monotone on purpose) ==")
    rc_s, out_s = run("--geometry", "--stack", "--frac", "0.3")
    print("\n".join("   " + l for l in out_s.strip().splitlines()[-9:]))
    check(rc_s == 0, "project_q36 --stack renders")
    row = {}
    for line in out_s.splitlines():
        for label in ("F16 trunk", "int4 trunk", "+ layer schedule",
                      "+ window cache", "+ speculation"):
            if line.startswith(label):
                row[label] = float(line[len(label):].split()[2])
    need = ("F16 trunk", "int4 trunk", "+ layer schedule", "+ window cache", "+ speculation")
    check(all(k in row for k in need), "all five stack rows present", f"{sorted(row)}")
    if all(k in row for k in need):
        check(row["int4 trunk"] < row["F16 trunk"] * 0.25,
              "int4 trunk cuts the modelled s/token >4x",
              f"{row['F16 trunk']:.0f} -> {row['int4 trunk']:.0f} s/token")
        check(row["+ layer schedule"] <= row["int4 trunk"],
              "the layer schedule never costs more than a flat frac at equal bytes",
              f"{row['int4 trunk']:.1f} -> {row['+ layer schedule']:.1f} s/token")
        check(row["+ window cache"] <= row["+ layer schedule"] + 1e-6,
              "the window never increases modelled cost",
              f"{row['+ layer schedule']:.1f} -> {row['+ window cache']:.1f} s/token")
        check(row["+ speculation"] < row["+ window cache"] / 2.0,
              "speculation multiplies the whole stack (2.86 accepted/forward, measured)",
              f"{row['+ window cache']:.1f} -> {row['+ speculation']:.1f} s/token")
    print("\n== the window's capacity is the ENGINE's formula, not a wish ==")
    rc_w, out_w = run("--geometry", "--trunk-dtype", "I4", "--small-dtype", "I4",
                      "--attn-dtype", "I4", "--win-mb", "512", "--frac", "0.25")
    rows_per_layer = grab(out_w, r"window Q36_WIN_MB=512: ([\d.]+) rows/layer")
    # engine: per-slot = 3*(hidden/2)+16 = 7696 B, split over 64 MLP layers
    expect = 512 * 1024 * 1024 / 64 / 7696
    check(rows_per_layer is not None and abs(rows_per_layer - expect) < 2.0,
          "modelled rows/layer match q36_win_ensure's arithmetic",
          f"model {rows_per_layer} vs engine formula {expect:.1f}")
    rc_small, out_small = run("--geometry", "--trunk-dtype", "I4", "--small-dtype", "I4",
                              "--attn-dtype", "I4", "--win-mb", "4", "--frac", "0.25")
    check("64-rows/layer floor" in out_small,
          "a below-floor budget is reported as 'would not allocate' (the E14 signature)")
    print("\n== --floor: the floor analysis must be arithmetic, not narrative ==")
    if floor_gate():
        FAILS.append("floor")

    print(f"\n{'PROJECTION GATE PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
