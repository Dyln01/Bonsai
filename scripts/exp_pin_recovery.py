#!/usr/bin/env python3
"""exp_pin_recovery.py v2 — learned hot-store (pin) benefit, with a REAL history.

REPLAY mode does not append .coli_usage (that happens per-turn in serve mode),
so this script first plays a synthetic history into <model>/.coli_usage via
SERVE-mode turns (each turn appends), then measures:
  arm1 cold : COLI_USAGE="" (ignore history)      -> baseline
  arm2 pin  : PIN=auto (seed hot store from .coli_usage) + PIN_GB
The engine's [PIN] lines and hit-rate/tok-s from REPLAY decode are the readout.
"""
import json, os, re, subprocess, time

ENGINE = os.path.expanduser("~/colibri/c/colibri")
MODEL = "/home/dylan/synth_models/synth_std"
RE_REPLAY = re.compile(r"REPLAY decode: (\d+) tokens in ([\d.]+)s \| ([\d.]+) tok/s \| expert hit ([\d.]+)%")
RE_RES = re.compile(r"resident experts: (\d+) pinned \(([\d.]+) GB\) \+ (\d+) in LRU \(([\d.]+) GB")
RE_PIN = re.compile(r"\[PIN\].*")

def serve_turns(n_turns=12, prompt="please explain the nature of synthetic model testing in great detail"):
    """Drive SERVE mode with n turns so .coli_usage accumulates selections."""
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("IOTH_", "SNAP", "REF", "REPLAY", "CTX", "KVSAVE", "COLI_USAGE",
                         "USAGE_SAVE", "PIN", "AUTOPIN", "PIN_GB", "SERVE", "PROMPT")):
            del e[k]
    e.update({"SNAP": MODEL, "CTX": "2048", "KVSAVE": "0", "USAGE_SAVE": "1",
              "CHAT_TEMPLATE": "0", "NGEN": "64", "COLI_TEMP": "0", "SERVE": "1",
              "CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1"})
    stdin = ((prompt + "\n") * n_turns).encode()
    p = subprocess.run([ENGINE, "8", "16", "16"], env=e, input=stdin,
                       capture_output=True, timeout=3600)
    p.stdout = p.stdout.decode(errors="replace"); p.stderr = p.stderr.decode(errors="replace")
    return p

def replay_run(pin=False, pin_gb="1.5"):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("IOTH_", "SNAP", "REF", "REPLAY", "CTX", "KVSAVE", "COLI_USAGE",
                         "USAGE_SAVE", "PIN", "AUTOPIN", "PIN_GB", "SERVE", "PROMPT")):
            del e[k]
    e.update({"SNAP": MODEL, "REF": f"{MODEL}/ref_replay.json", "REPLAY": "1",
              "NGEN": "512", "CTX": "1024", "KVSAVE": "0", "CAP": "8",
              "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1", "COLI_TEMP": "0",
              "USAGE_SAVE": "0"})
    if pin:
        e["PIN"] = "auto"; e["PIN_GB"] = pin_gb
    else:
        # true cold arm: point the usage path at a nonexistent file so
        # usage_load() sees zero history and pins nothing (COLI_USAGE=""
        # would fall back to <snap>/.coli_usage and pin anyway)
        e["COLI_USAGE"] = "/tmp/nonexistent_usage_4912.txt"
    p = subprocess.run([ENGINE, "8", "16", "16"], env=e, capture_output=True,
                       text=True, timeout=2400)
    m = RE_REPLAY.search(p.stdout)
    r = RE_RES.search(p.stdout)
    pins = RE_PIN.findall(p.stderr)
    out = {"rc": p.returncode}
    if m: out.update({"tok_s": float(m.group(3)), "hit_pct": float(m.group(4))})
    if r: out.update({"pin_n": int(r.group(1)), "pin_gb_res": float(r.group(2)),
                      "lru_n": int(r.group(3)), "lru_gb": float(r.group(4))})
    if pins: out["pin_lines"] = [l.strip() for l in pins][:3]
    return out

def main():
    results = {}
    print("== seeding .coli_usage via 12 serve turns ==")
    p = serve_turns(12)
    usage = os.path.join(MODEL, ".coli_usage")
    print(f"  .coli_usage: {os.path.exists(usage)}",
          f"{os.path.getsize(usage)} bytes" if os.path.exists(usage) else "")
    # AUTOPIN reads <snap>/.coli_usage directly (COLI_USAGE does not redirect it
    # in the colibri engine), so the true cold arm needs the file moved aside.
    hidden = usage + ".aside"
    os.replace(usage, hidden)
    try:
        print("== arm1: cold (history moved aside, no pin possible) ==")
        r = replay_run(pin=False); results["cold"] = r; print(" ", r)
    finally:
        os.replace(hidden, usage)
    print("== arm2: PIN=auto seeded from history ==")
    r = replay_run(pin=True); results["pin_auto"] = r; print(" ", r)
    with open(os.path.expanduser("~/colibri-research/harness/results_pin.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results_pin.json")

if __name__ == "__main__":
    main()
