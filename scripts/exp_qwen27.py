#!/usr/bin/env python3
"""exp_qwen27.py — Qwen-shaped dense model: long-context KV cost.

Qwen3.8-27B architecture (scaled): GQA K/V caches are FULL head_dim f32
(no MLA latent compression in this family), so KV RAM = 2 * L * ctx * kv_heads
* head_dim * 4 B. At real-27B scale (64 layers, kv_heads 8, head_dim 128):
  262144 ctx -> 2*64*262144*8*128*4 = 34.4 GB f32.
The qwen36 engine has NO KV8/KV_TQ path (grep: 0 hits) — this experiment
measures what the KV actually costs in RSS and decode speed at long ctx, which
quantifies the opportunity for porting the GLM engine's KV quantization here.
"""
import os, re, subprocess, time
_Q36_HOME = os.environ.get("Q36_HOME", "/home/dylan")  # portability: override per host

ENGINE = os.path.expanduser("~/colibri/c/qwen36")
MODEL = f"{_Q36_HOME}/synth_models/synth_qwen27"
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_TOKS = re.compile(r"\[TOKENS\] (\d+) generated:(.*)$", re.M)
RE_GEN = re.compile(r"gen: (\d+) tokens in ([\d.]+)s \| ([\d.]+) tok/s", re.M)

def run(ctx, ngen=64, kv_bytes_note=""):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("SNAP", "CTX", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "PROMPT", "IOTH_")):
            del e[k]
    e.update({"SNAP": MODEL, "CTX": str(ctx), "NGEN": str(ngen), "KVSAVE": "0",
              "COLI_USAGE": "/tmp/none", "USAGE_SAVE": "0", "COLI_TEMP": "0",
              "TOKENS": "1"})
    # qwen36 reads its prompt from a FILE (argv[3], non-.json = raw text)
    pt = f"{_Q36_HOME}/prompt_test.txt"
    open(pt, "w").write("the quick brown fox jumps over the lazy dog and runs far away "
                        "while the sun sets slowly over the mountains and rivers of the valley below")
    cmd = ["/usr/bin/time", "-v", ENGINE, "32", "8", pt]
    t0 = time.time()
    p = subprocess.run(cmd, env=e, capture_output=True, timeout=3600)
    wall = time.time() - t0
    p.stdout = p.stdout.decode(errors="replace")
    p.stderr = p.stderr.decode(errors="replace")
    rss = None
    m = RE_RSS.search(p.stderr)
    if m: rss = round(int(m.group(1)) / 1024, 1)
    g = RE_GEN.search(p.stderr) or RE_GEN.search(p.stdout)
    toks = None
    if not g:
        m2 = RE_TOKS.search(p.stderr)
        if m2: toks = len(m2.group(2).split())
    return {"ctx": ctx, "peak_rss_mb": rss, "rc": p.returncode,
            "gen_line": g.group(0) if g else None, "tokens_dumped": toks,
            "wall_s": round(wall, 1)}

def main():
    # synthetic: KV bytes/tok = 2 * L * KVH * HD * 4 = 2*24*4*64*4 = 49,152 B
    # ctx 1024 -> 48 MB | 8192 -> 384 MB | 32768 -> 1.5 GB | 131072 -> 6 GB
    for ctx in [1024, 8192, 32768]:
        r = run(ctx)
        # theoretical KV
        kv_mb = 2 * 24 * ctx * 4 * 64 * 4 / 1e6
        print(f"ctx={ctx:6d}: rc={r['rc']} peak_rss={r['peak_rss_mb']} MB "
              f"(theoretical KV {kv_mb:.0f} MB) wall={r['wall_s']}s {r['gen_line'] or ''}")
    print("\n-- real 27B scale extrapolation (64 L, kv_heads 8, head_dim 128): --")
    for ctx in [32768, 65536, 262144]:
        kv_gb = 2 * 64 * ctx * 8 * 128 * 4 / 1e9
        print(f"ctx={ctx:6d}: f32 KV = {kv_gb:.1f} GB  | with an 8-bit KV port: {kv_gb/4:.1f} GB | 4-bit: {kv_gb/8:.1f} GB")

if __name__ == "__main__":
    main()
