#!/usr/bin/env python3
"""exp_mtp_depth_q36.py — measured tokens/forward vs MTP draft depth.

The spec machinery already reports everything this needs, per run, on stderr:

  [qwen36 lookup] <scope>: X.XX tokens/forward (N forwards per M tokens) |
                  acceptance P% (a/b drafts in c verifies) | gate on/off, d declined, p probes

so "how much does one more draft buy" is a sweep of an EXISTING knob (Q36_MTP_DRAFTS=0..3)
plus a parse — no engine change. This is the measurement half of the MTP sketch; the
generative half (tree / multi-candidate drafts) is NOT implemented and is not claimed here.

    python3 scripts/exp_mtp_depth_q36.py <model_dir> <prompt_file> [--depths 0,1,2,3]
                                        [--nnew 64] [--out results/mtp_depth_<host>.json]

Needs a container that CARRIES an MTP head (scripts/gen_mtp_head_synth.py adds one to a
synthetic; a real checkpoint ships its own). Without one every arm is a skipped arm and the
script says so instead of inventing numbers. On random-weight heads acceptance is ~0 by
construction (the house rule: synthetics are mechanical oracles), so real acceptance is a
real-weights measurement — the field's own M1 arm measured 82.0% / 2.86 tokens/forward.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

LOOKUP = re.compile(r"\[qwen36 lookup\] (\S+): ([\d.]+) tokens/forward \((\d+) forwards per "
                    r"(\d+) tokens\) \| acceptance ([\d.]+)% \((\d+)/(\d+) drafts in (\d+) "
                    r"verifies\) \| gate (\w+)(.*)")
MTPRUN = re.compile(r"\[qwen36 MTP\] run:(.*)")


def parse(err_text):
    """Last 'lookup' line of a run -> its numbers. The engine prints one per run scope."""
    out = None
    for m in LOOKUP.finditer(err_text):
        out = dict(scope=m.group(1), tokens_per_forward=float(m.group(2)),
                   forwards=int(m.group(3)), tokens=int(m.group(4)),
                   acceptance_pct=float(m.group(5)), accepted=int(m.group(6)),
                   drafts=int(m.group(7)), verifies=int(m.group(8)), gate=m.group(9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("--depths", default="0,1,2,3")
    ap.add_argument("--nnew", type=int, default=64)
    ap.add_argument("--maxt", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=0, help="per-arm seconds (0 = none)")
    ap.add_argument("--engine", default=os.path.expanduser("~/colibri/c/qwen36"))
    ap.add_argument("--out")
    a = ap.parse_args()

    meta_p = os.path.join(a.model, "qwen36_meta.json")
    if not os.path.exists(meta_p):
        sys.exit(f"{a.model}: no qwen36_meta.json")
    meta = json.load(open(meta_p))
    if not meta.get("mtp"):
        sys.exit(f"{a.model}: container carries no MTP head (meta.mtp missing). Add one to a "
                 f"synthetic with scripts/gen_mtp_head_synth.py, or use a checkpoint that "
                 f"ships one — without a head every arm here is a skip, not a measurement.")
    depths = [int(x) for x in a.depths.split(",") if x.strip() != ""]
    if not os.path.exists(a.prompt):
        sys.exit(f"{a.prompt}: prompt file not found")
    if not os.path.exists(a.engine):
        sys.exit(f"{a.engine}: engine binary not found (build: make -C ~/colibri/c qwen36)")

    rows = []
    print(f"== MTP draft depth sweep: {a.model}\n   depths {depths}, n_new {a.nnew}, "
          f"threads {a.threads}, engine {a.engine}")
    print(f"\n{'drafts':>7} {'tok/forward':>12} {'acceptance':>11} {'drafts':>7} "
          f"{'verifies':>9} {'wall s':>8} {'effective tps':>14}  note")
    base_tps = None
    for d in depths:
        env = dict(os.environ)
        env.update(SNAP=a.model, TRUNK_MAP="1", Q36_MAXT=str(a.maxt), N_NEW=str(a.nnew),
                   COLI_TEMP="0", OMP_NUM_THREADS=str(a.threads), COLI_NO_OMP_TUNE="1",
                   KVSAVE="0", Q36_MTP="1", Q36_MTP_DRAFTS=str(d), COLI_TIMERS="1")
        cmd = [a.engine, "8", str(a.threads), a.prompt]
        if a.timeout:
            cmd = ["timeout", str(a.timeout)] + cmd
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.time() - t0
        err = p.stderr
        num = parse(err)
        note = []
        if p.returncode != 0:
            note.append(f"rc={p.returncode}" + (" (wall kill)" if p.returncode == 124 else ""))
        if num is None:
            note.append("no [qwen36 lookup] line — one decode step? raise --nnew")
            rows.append(dict(depth=d, rc=p.returncode, wall_s=round(wall, 1),
                             parse=None, note="; ".join(note)))
            print(f"{d:7d} {'-':>12} {'-':>11} {'-':>7} {'-':>9} {wall:8.1f} "
                  f"{'-':>14}  {'; '.join(note)}")
            continue
        ms = None
        m = re.search(r"\[timers\]\s+TOTAL\s+([\d.]+) ms\s+([\d.]+) ms/token", err)
        if m:
            ms = float(m.group(2))
        tps = (1000.0 / ms) if ms else None
        eff = (float(num["tokens_per_forward"]) * tps) if (tps and num) else None
        if d == 0 and eff:
            base_tps = eff
        if base_tps and eff:
            note.append(f"{eff / base_tps:.2f}x vs drafts=0")
        # `depth` is the knob we swept; `drafts` (inside **num) is the engine's own
        # count of drafts it actually issued — they collide on purpose, so keep
        # the knob named depth.
        rows.append(dict(depth=d, rc=p.returncode, wall_s=round(wall, 1), **num,
                         ms_per_token=ms, decode_tps=tps, effective_tps=eff,
                         note="; ".join(note)))
        print(f"{d:7d} {num['tokens_per_forward']:12.2f} {num['acceptance_pct']:10.1f}% "
              f"{num['drafts']:7d} {num['verifies']:9d} {wall:8.1f} "
              f"{(f'{eff:.2f}' if eff else '-'):>14}  {'; '.join(note)}")

    good = [r for r in rows if r.get("effective_tps")]
    print()
    if len(good) < 2:
        print("Not enough parsed arms for a verdict (see the notes above).")
    else:
        best = max(good, key=lambda r: r["effective_tps"])
        print(f"best measured: Q36_MTP_DRAFTS={best['depth']} -> "
              f"{best['tokens_per_forward']:.2f} tokens/forward "
              f"({best['drafts']} drafts issued) at {best['acceptance_pct']:.1f}% acceptance "
              f"= {best['effective_tps']:.2f} tok/s effective")
        print("Acceptance is a property of the HEAD and the text, not of this script: on "
              "random-weight containers expect ~0 (mechanical oracle only), and quote a real "
              "number only from a real-weights run.")
    out = a.out or f"results/mtp_depth_{platform.node()}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(dict(model=os.path.abspath(a.model), prompt=os.path.abspath(a.prompt),
                   n_new=a.nnew, maxt=a.maxt, threads=a.threads, depths=depths,
                   arms=rows), open(out, "w"), indent=1)
    print(f"json: {out}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
