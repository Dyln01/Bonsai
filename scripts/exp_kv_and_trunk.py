import os
#!/usr/bin/env python3
"""exp_kv_and_trunk.py — focused experiments for research areas #1 (KV RAM)
and #2 (dense-trunk offload), on the synthetic harness.

Experiment A — KV cache RAM at long context (PROMPT mode, CTX=131072):
  In generate mode the engine allocates the KV cache for the full CTX up
  front, so a short prompt + CTX=131072 isolates KV RAM cleanly. Arms:
    f32 (default) vs KV8=1 vs KV_TQ=4. Measure peak RSS.
  NOTE: KV8/KV_TQ change KV precision (documented lossy knobs); the RSS
  delta is the measurement, not an endorsement of default-on.

Experiment B — dense-trunk offload (synth_huge_trunk, trunk ~0.77 GB):
  RAM_GB=2 forces a small expert budget. Arms:
    B1 trunk resident (default) — the baseline the proposal wants to fix;
    B2 TRUNK_RESIDENT_LAYERS=0 + COLI_MMAP=1 — trunk paged from disk.
  Measure peak RSS, tok/s, and whether either arm refuses/OOMs.

Experiment C — determinism gates (verify_structural.py inline): greedy token
  stream hash x2 fresh processes must match.
"""
import hashlib, json, os, re, subprocess, sys, time
_Q36_HOME = os.environ.get("Q36_HOME", "/home/dylan")  # portability: override per host

ENGINE = os.path.expanduser("~/colibri/c/colibri")
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_TOKS = re.compile(r"\[TOKENS\] (\d+) generated:(.*)$", re.M)
RE_REPLAY = re.compile(r"REPLAY decode: (\d+) tokens in ([\d.]+)s \| ([\d.]+) tok/s \| expert hit ([\d.]+)%")

def run(env_extra, cap="64"):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("IOTH_", "SNAP", "REF", "REPLAY", "CTX", "KVSAVE", "COLI_USAGE", "USAGE_SAVE")):
            del e[k]
    e.update(env_extra)
    cmd = ["/usr/bin/time", "-v", ENGINE, cap, "16", "16"]
    return subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=1800)

def peak_rss_mb(se):
    m = RE_RSS.search(se)
    return round(int(m.group(1)) / 1024, 1) if m else None

def main():
    out = {}
    print("== A: KV RAM at CTX=131072 (synth_std, PROMPT mode) ==")
    for label, kv in [("f32", {}), ("kv8", {"KV8": "1"}), ("kvtq4", {"KV_TQ": "4"})]:
        env = {"SNAP": f"{_Q36_HOME}/synth_models/synth_std", "CTX": "131072",
               "NGEN": "8", "KVSAVE": "0", "COLI_USAGE": "", "USAGE_SAVE": "0",
               "COLI_TEMP": "0", "CHAT_TEMPLATE": "0", "CAP_RAISE": "0",
               "PROMPT": "the quick brown fox jumps over the lazy dog repeatedly"}
        env.update(kv)
        p = run(env, cap="8")
        rss = peak_rss_mb(p.stderr)
        out[f"kv_131072_{label}"] = {"peak_rss_mb": rss, "rc": p.returncode}
        print(f"  {label}: rc={p.returncode} peak_rss={rss} MB")
        if p.returncode != 0:
            print("   stderr:", p.stderr[-300:].replace("\n", " | "))

    print("== B: dense-trunk offload (synth_huge_trunk, trunk ~0.77 GB, RAM_GB=2) ==")
    for label, tr in [("resident_default", {}),
                      ("trunk_mmap0", {"TRUNK_RESIDENT_LAYERS": "0", "COLI_MMAP": "1"}),
                      ("trunk_mmap0_pilot", {"TRUNK_RESIDENT_LAYERS": "0", "COLI_MMAP": "1", "PIPE": "1"})]:
        env = {"SNAP": f"{_Q36_HOME}/synth_models/synth_huge_trunk", "CTX": "2048",
               "NGEN": "16", "KVSAVE": "0", "COLI_USAGE": "", "USAGE_SAVE": "0",
               "COLI_TEMP": "0", "CHAT_TEMPLATE": "0", "RAM_GB": "2",
               "CAP_RAISE": "0", "DIRECT": "1",
               "PROMPT": "the quick brown fox jumps over the lazy dog repeatedly"}
        env.update(tr)
        p = run(env, cap="4")
        rss = peak_rss_mb(p.stderr)
        out[f"trunk_{label}"] = {"peak_rss_mb": rss, "rc": p.returncode}
        print(f"  {label}: rc={p.returncode} peak_rss={rss} MB")
        if p.returncode != 0:
            print("   stderr:", p.stderr[-300:].replace("\n", " | "))

    print("== C: greedy determinism (2 fresh processes, synth_std) ==")
    hashes = []
    for i in range(2):
        env = {"SNAP": f"{_Q36_HOME}/synth_models/synth_std", "CTX": "1024",
               "NGEN": "64", "KVSAVE": "0", "COLI_USAGE": "", "USAGE_SAVE": "0",
               "COLI_TEMP": "0", "CHAT_TEMPLATE": "0", "TOKENS": "1",
               "PROMPT": "the quick brown fox jumps over the lazy dog repeatedly and considers the nature of testing synthetic models"}
        p = run(env, cap="16")
        m = RE_TOKS.search(p.stderr)
        if not m:
            print(f"  run {i}: no [TOKENS]; rc={p.returncode}")
            print("   stderr:", p.stderr[-200:].replace("\n", " | "))
            continue
        h = hashlib.sha256(m.group(2).encode()).hexdigest()
        hashes.append(h)
        print(f"  run {i}: sha256 {h[:16]}… ({len(m.group(2).split())} tokens)")
    out["determinism_pass"] = len(hashes) == 2 and hashes[0] == hashes[1]
    print(f"  determinism: {'PASS' if out['determinism_pass'] else 'FAIL'}")

    with open(os.path.expanduser("~/colibri-research/harness/results_kv_trunk.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results_kv_trunk.json")

if __name__ == "__main__":
    main()
