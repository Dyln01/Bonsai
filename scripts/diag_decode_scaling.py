#!/usr/bin/env python3
"""diag_decode_scaling.py — find where decode time actually goes.

Runs decode at several context lengths with:
  - a NON-repetitive prompt (random token ids — no n-gram pathology)
  - COLI_LOOKUP on vs off (the speculation source)
  - 8 vs 16 OMP threads
and prints tok/s per arm. The prompt file format: one token id per... the
engine reads raw text; distinct words = distinct byte-level tokens.
"""
import subprocess, os, re, random

E = os.path.expanduser("~/colibri/c/qwen36")
RE_SPEED = re.compile(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)")
RE_TTFT = re.compile(r"TTFT: ([\d.]+) s")
RE_SPEC = re.compile(r"\[qwen36 lookup\] run: ([\d.]+) tokens/forward \((\d+) forwards")

def mkprompt(n_words, path):
    # distinct 3-char pseudo-words -> distinct byte tokens, no repeats
    rng = random.Random(99)
    words = []
    for i in range(n_words):
        words.append("w%d" % i)
    open(path, "w").write(" ".join(words))

def run(ctx, lookup, threads, prompt_file, ngen=64):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("SNAP", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "Q36_MAXT",
                         "NGEN", "COLI_TEMP", "KV8", "COLI_DENSE_BITS", "COLI_LOOKUP",
                         "OMP_NUM_THREADS", "COLI_NO_OMP_TUNE")):
            del e[k]
    e.update({"SNAP": "/home/dylan/synth_models/synth_qwen27", "KVSAVE": "0",
              "COLI_USAGE": "/tmp/none", "USAGE_SAVE": "0", "Q36_MAXT": "16384",
              "NGEN": str(ngen), "COLI_TEMP": "0", "OMP_NUM_THREADS": str(threads),
              "COLI_NO_OMP_TUNE": "1"})
    if lookup:
        e["COLI_LOOKUP"] = "1"
    else:
        e["COLI_LOOKUP"] = "0"
    p = subprocess.run(["/usr/bin/time", "-v", E, "32", "8", prompt_file],
                       env=e, capture_output=True)
    err = p.stderr.decode(errors="replace")
    sp = RE_SPEED.search(err)
    tt = RE_TTFT.search(err)
    spec = RE_SPEC.search(err)
    return (float(sp.group(1)) if sp else None,
            float(tt.group(1)) if tt else None,
            spec.group(0) if spec else None,
            p.returncode)

def main():
    # distinct-token prompts at several lengths
    for n in [512, 2048, 8192]:
        pf = f"/home/dylan/prompt_distinct_{n}.txt"
        mkprompt(n, pf)
        for lookup in [False, True]:
            for th in [8]:
                toks, ttft, spec, rc = run(16384 if False else n + 128, lookup, th, pf)
                print(f"ctx~{n:5d} words | LOOKUP={'1' if lookup else '0'} | threads={th}: "
                      f"decode={toks} tok/s | ttft={ttft}s | rc={rc}")
                if spec: print(f"    {spec}")
    print("\n-- 16 threads at 8k, lookup off --")
    pf = "/home/dylan/prompt_distinct_8192.txt"
    toks, ttft, spec, rc = run(8192 + 128, False, 16, pf)
    print(f"threads=16: decode={toks} tok/s ttft={ttft}s rc={rc}")

if __name__ == "__main__":
    main()
