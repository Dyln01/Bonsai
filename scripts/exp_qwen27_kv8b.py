import os
#!/usr/bin/env python3
"""exp_qwen27_kv8b.py — medium-context (8k) f32 vs KV8 comparison, fixed quoting."""
import subprocess, os, re
_Q36_HOME = os.environ.get("Q36_HOME", "/home/dylan")  # portability: override per host

E = os.path.expanduser("~/colibri/c/qwen36")
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_SPEED = re.compile(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)")
RE_TTFT = re.compile(r"TTFT: ([\d.]+) s")

txt = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 280
open(f"{_Q36_HOME}/prompt_8k.txt", "w").write(txt)

for kv8 in [False, True]:
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("SNAP", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "Q36_MAXT", "NGEN", "COLI_TEMP", "KV8")):
            del e[k]
    e.update({"SNAP": f"{_Q36_HOME}/synth_models/synth_qwen27", "KVSAVE": "0",
              "COLI_USAGE": "/tmp/none", "USAGE_SAVE": "0", "Q36_MAXT": "8192",
              "NGEN": "48", "COLI_TEMP": "0"})
    if kv8:
        e["KV8"] = "1"
    p = subprocess.run(["/usr/bin/time", "-v", E, "32", "8", f"{_Q36_HOME}/prompt_8k.txt"],
                       env=e, capture_output=True)
    err = p.stderr.decode(errors="replace")
    rss = RE_RSS.search(err)
    sp = RE_SPEED.search(err)
    tt = RE_TTFT.search(err)
    label = "kv8" if kv8 else "f32"
    r = int(rss.group(1)) / 1024 if rss else -1
    t = tt.group(1) if tt else "?"
    s = sp.group(1) if sp else "?"
    print(f"{label}: rss={r:.0f}MB ttft={t}s decode={s} tok/s rc={p.returncode}")
