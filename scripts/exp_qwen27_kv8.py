#!/usr/bin/env python3
"""exp_qwen27_kv8.py — measure the KV8-patched qwen36 vs f32 at long context.

Method: a ~34k-token prefill actually touches the KV rows (unlike short runs
where lazy calloc pages cost ~0 RSS). Compare peak RSS f32 vs KV8 and the
per-token decode speed after the long prefill. Also records prefill time.
"""
import os, re, subprocess, sys, time
_Q36_HOME = os.environ.get("Q36_HOME", "/home/dylan")  # portability: override per host

ENGINE = os.path.expanduser("~/colibri/c/qwen36")
MODEL = f"{_Q36_HOME}/synth_models/synth_qwen27"
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_SPEED = re.compile(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)")
RE_TTFT = re.compile(r"TTFT: ([\d.]+) s")

def run(kv8, prompt_file, ngen=48):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("SNAP", "CTX", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "Q36_MAXT", "NGEN", "COLI_TEMP", "KV8")):
            del e[k]
    e.update({"SNAP": MODEL, "KVSAVE": "0", "COLI_USAGE": "/tmp/none", "USAGE_SAVE": "0",
              "Q36_MAXT": "40960", "NGEN": str(ngen), "COLI_TEMP": "0"})
    if kv8:
        e["KV8"] = "1"
    cmd = ["/usr/bin/time", "-v", ENGINE, "32", "8", prompt_file]
    t0 = time.time()
    p = subprocess.run(cmd, env=e, capture_output=True, timeout=3600)
    wall = time.time() - t0
    err = p.stderr.decode(errors="replace")
    rss = None
    m = RE_RSS.search(err)
    if m: rss = round(int(m.group(1)) / 1024, 1)
    sp = RE_SPEED.search(err)
    tt = RE_TTFT.search(err)
    return {"kv8": kv8, "peak_rss_mb": rss, "rc": p.returncode, "wall_s": round(wall, 1),
            "tok_s": float(sp.group(1)) if sp else None,
            "ttft_s": float(tt.group(1)) if tt else None}

def main():
    pf = f"{_Q36_HOME}/prompt_long.txt"
    if not os.path.exists(pf):
        open(pf, "w").write("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 1200)
    results = {}
    for kv8 in [False, True]:
        r = run(kv8, pf)
        key = "kv8" if kv8 else "f32"
        results[key] = r
        print(f"{key:4s}: rc={r['rc']} peak_rss={r['peak_rss_mb']} MB | decode {r['tok_s']} tok/s | TTFT {r['ttft_s']}s | wall {r['wall_s']}s")
    if results["f32"]["peak_rss_mb"] and results["kv8"]["peak_rss_mb"]:
        d = results["f32"]["peak_rss_mb"] - results["kv8"]["peak_rss_mb"]
        print(f"\nKV RAM saved by KV8 at ~34k ctx: {d:.0f} MB "
              f"({100*d/results['f32']['peak_rss_mb']:.0f}% of peak RSS)")
    with open(os.path.expanduser("~/colibri-research/harness/results_qwen_kv8.json"), "w") as f:
        import json
        json.dump(results, f, indent=2)
    print("wrote results_qwen_kv8.json")

if __name__ == "__main__":
    main()
