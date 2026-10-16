#!/usr/bin/env python3
"""exp_kv_serve.py — KV-cache RAM at long context, via SERVE mode.

The engine allocates the full KV pool for CTX in serve mode (kv_alloc(maxctx)
per slot), so this isolates KV RAM cleanly:
    PROMPT-mode KV is sized by actual tokens; serve mode pre-allocates CTX.

Arms: f32 (default) vs KV8=1 vs KV_TQ=4, at CTX=131072, KV_SLOTS=1.
Feeds one short line over the byte protocol, reads STAT lines, then \x02RESET
and EOF; peak RSS from /usr/bin/time -v wraps the whole server process.

Also measures SERVE decode tok/s with KV8/KV_TQ vs f32 (the CPU attention
path dequantizes per tile — the throughput cost of KV quantization).
"""
import json, os, re, subprocess, sys, time

ENGINE = os.path.expanduser("~/colibri/c/colibri")
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_STAT = re.compile(r"^STAT (\d+) ([\d.]+) ([\d.]+) ([\d.]+)", re.M)
RE_POOL = re.compile(r"\[KV\] context slots: (\d+) x (\d+) tokens, projected pool ([\d.]+) GB")


def serve_run(model, ctx, env_extra, turns=2, ngen=48, prompt="hello world, tell me about the weather and the sea and the ships"):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("IOTH_", "SNAP", "REF", "REPLAY", "CTX", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "PROMPT")):
            del e[k]
    e.update(env_extra)
    e.update({"SNAP": model, "CTX": str(ctx), "KVSAVE": "0", "COLI_USAGE": "",
              "USAGE_SAVE": "0", "CHAT_TEMPLATE": "0", "NGEN": str(ngen), "COLI_TEMP": "0",
              "SERVE": "1"})
    cmd = ["/usr/bin/time", "-v", ENGINE, "8", "16", "16"]
    t0 = time.time()
    # protocol: send one line, wait for END sentinel, then close (EOF exits)
    stdin_data = (prompt + "\n").encode()
    p = subprocess.run(cmd, env=e, input=stdin_data, capture_output=True, timeout=3600)
    wall = time.time() - t0
    # decode: input was bytes, so outputs come back as bytes
    p.stdout = p.stdout.decode(errors="replace")
    p.stderr = p.stderr.decode(errors="replace")
    return p, wall


def main():
    model = "/home/dylan/synth_models/synth_std"
    out = {}
    print("== KV RAM at CTX=131072 via SERVE mode (KV pool pre-allocated) ==")
    for label, kv in [("f32", {}), ("kv8", {"KV8": "1"}), ("kvtq4", {"KV_TQ": "4"})]:
        try:
            p, wall = serve_run(model, 131072, kv)
        except subprocess.TimeoutExpired:
            out[label] = {"rc": "timeout"}
            print(f"  {label}: TIMEOUT")
            continue
        rss = None
        m = RE_RSS.search(p.stderr)
        if m: rss = round(int(m.group(1)) / 1024, 1)
        pool = None
        m = RE_POOL.search(p.stderr)
        if m: pool = float(m.group(3))
        stats = RE_STAT.findall(p.stdout)
        toks = sum(int(s[0]) for s in stats) if stats else 0
        rate = sum(float(s[1]) for s in stats) if stats else 0.0
        out[label] = {"peak_rss_mb": rss, "projected_pool_gb": pool, "rc": p.returncode,
                      "tokens": toks, "tok_s_avg": round(rate / len(stats), 2) if stats else None}
        print(f"  {label}: rc={p.returncode} peak_rss={rss} MB | pool={pool} GB | {toks} tokens in {len(stats)} turn(s)")
        if p.returncode != 0:
            print("   stderr tail:", p.stderr[-300:].replace("\n", " | "))
    # short-context control: same arms at CTX=2048 (pool should be ~64x smaller)
    print("== control: CTX=2048 ==")
    for label, kv in [("f32", {}), ("kv8", {"KV8": "1"}), ("kvtq4", {"KV_TQ": "4"})]:
        p, wall = serve_run(model, 2048, kv)
        rss = None
        m = RE_RSS.search(p.stderr)
        if m: rss = round(int(m.group(1)) / 1024, 1)
        out[f"ctx2048_{label}"] = rss
        print(f"  {label}: peak_rss={rss} MB")
    with open(os.path.expanduser("~/colibri-research/harness/results_kv_serve.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results_kv_serve.json")

if __name__ == "__main__":
    main()
