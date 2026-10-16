#!/usr/bin/env python3
"""exp_qwen27_final.py — the full low-RAM/high-speed battery for qwen36.

Arms (synthetic Qwen3.8-27B-shaped container, ~8k-token context):
  A. f32 KV + int8 dense (engine defaults)          — baseline
  B. KV8 (v2 bit-math decoder)                      — RAM: -KV; speed: ?
  C. f32 KV + COLI_DENSE_BITS=4 (int4 dense)        — RAM: -dense; speed: +matmul
  D. KV8 + COLI_DENSE_BITS=4                        — the low-RAM profile: both
Records peak RSS, TTFT, decode tok/s. The point: find whether the KV8-v2
decoder closed the 1.7x gap, and quantify the combined RAM/Speed point that
a "worse hardware" deployment would run.
"""
import subprocess, os, re, json

E = os.path.expanduser("~/colibri/c/qwen36")
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_SPEED = re.compile(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)")
RE_TTFT = re.compile(r"TTFT: ([\d.]+) s")

def run(env_extra, prompt_file):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("SNAP", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "Q36_MAXT",
                         "NGEN", "COLI_TEMP", "KV8", "COLI_DENSE_BITS", "COLI_DENSE_INT4")):
            del e[k]
    e.update({"SNAP": "/home/dylan/synth_models/synth_qwen27", "KVSAVE": "0",
             "COLI_USAGE": "/tmp/none", "USAGE_SAVE": "0", "Q36_MAXT": "8192",
             "NGEN": "48", "COLI_TEMP": "0"})
    e.update(env_extra)
    p = subprocess.run(["/usr/bin/time", "-v", E, "32", "8", prompt_file],
                       env=e, capture_output=True)
    err = p.stderr.decode(errors="replace")
    rss = RE_RSS.search(err)
    sp = RE_SPEED.search(err)
    tt = RE_TTFT.search(err)
    return {"peak_rss_mb": round(int(rss.group(1)) / 1024, 1) if rss else None,
            "ttft_s": float(tt.group(1)) if tt else None,
            "decode_tok_s": float(sp.group(1)) if sp else None,
            "rc": p.returncode}

def main():
    pf = "/home/dylan/prompt_8k.txt"
    if not os.path.exists(pf):
        open(pf, "w").write("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 280)
    arms = {
        "A_defaults_f32kv_int8dense": {},
        "B_kv8_v2": {"KV8": "1"},
        "C_int4dense": {"COLI_DENSE_BITS": "4"},
        "D_kv8_int4dense": {"KV8": "1", "COLI_DENSE_BITS": "4"},
    }
    results = {}
    for label, extra in arms.items():
        r = run(extra, pf)
        results[label] = r
        print(f"{label:28s}: rss={r['peak_rss_mb']}MB ttft={r['ttft_s']}s decode={r['decode_tok_s']} tok/s rc={r['rc']}", flush=True)
    base = results["A_defaults_f32kv_int8dense"]
    for label, r in results.items():
        if label == "A_defaults_f32kv_int8dense" or not base["peak_rss_mb"] or not r["peak_rss_mb"]:
            continue
        dr = base["peak_rss_mb"] - r["peak_rss_mb"]
        print(f"  {label}: RAM {dr:+.0f} MB ({100*dr/base['peak_rss_mb']:+.0f}%)", flush=True)
    with open(os.path.expanduser("~/colibri-research/harness/results_qwen_final.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results_qwen_final.json")

if __name__ == "__main__":
    main()
