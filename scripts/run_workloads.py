#!/usr/bin/env python3
"""run_workloads.py — colibri synthetic-harness benchmark driver.

Drives the REAL colibri engine against a synthetic container under the three
workload profiles from the research brief, with the LD_PRELOAD I/O fidelity
layer emulating a real disk (bandwidth cap + per-read latency floor; the
synthetic container would otherwise live entirely in the page cache).

Recorded per run: tok/s, expert hit rate, GB read, MB/token, per-load latency
percentiles (p50/p90/p99/max), peak RSS (from /usr/bin/time -v), pin/LRU
residency, time shares, and the engine's own verdict line.

Workload profiles:
  cold    — small cache caps (64, 8 slots/layer) force heavy cold misses; the
            throttle layer adds real-disk latency+bandwidth so the I/O time
            is realistic even though the page cache is warm (a true cold page
            cache needs root, documented as a limitation).
  steady  — Zipf-routed REPLAY of a long mixed stream, PIPE=1 vs PIPE=0 to A/B
            read/compute overlap.
  longctx — REPLAY at CTX=131072 with the same stream, stressing KV RAM.

Usage:
  python3 run_workloads.py --model ~/synth_models/synth_std --out results.json
"""
import argparse, json, os, re, subprocess, sys, time

RE_REPLAY = re.compile(
    r"REPLAY decode: (\d+) tokens in ([\d.]+)s \| ([\d.]+) tok/s \| expert hit ([\d.]+)%")
RE_IO = re.compile(r"expert I/O: ([\d.]+) GB fetched \(([\d.]+) MB/token")
RE_LAT = re.compile(r"latency p50 ([\d.]+) ms \| p90 ([\d.]+) ms \| p99 ([\d.]+) ms \| max ([\d.]+) ms")
RE_RES = re.compile(r"resident experts: (\d+) pinned \(([\d.]+) GB\) \+ (\d+) in LRU \(([\d.]+) GB")
RE_SHARE = re.compile(r"time shares: expert-I/O (\d+)% \| expert-matmul (\d+)% \| attention (\d+)% \| lm_head (\d+)% \| other (\d+)%")
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_DISKCLASS = re.compile(r"DISK-CLASS \(recency split\): cold (\d+) x ([\d.]+) GB")
RE_LOADTOK = re.compile(r"\| ([\d.]+) loads/token")


def parse_profile(stdout):
    out = {}
    m = RE_REPLAY.search(stdout)
    if m:
        out["tokens"] = int(m.group(1)); out["decode_s"] = float(m.group(2))
        out["tok_s"] = float(m.group(3)); out["hit_pct"] = float(m.group(4))
    m = RE_IO.search(stdout)
    if m:
        out["gb_read"] = float(m.group(1)); out["mb_per_tok"] = float(m.group(2))
    m = RE_LOADTOK.search(stdout)
    if m:
        out["loads_per_tok"] = float(m.group(1))
    m = RE_LAT.search(stdout)
    if m:
        out["p50_ms"], out["p90_ms"], out["p99_ms"], out["max_ms"] = \
            (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    m = RE_RES.search(stdout)
    if m:
        out["pin_n"], out["pin_gb"], out["lru_n"], out["lru_gb"] = \
            int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))
    m = RE_SHARE.search(stdout)
    if m:
        out["share_io"], out["share_matmul"], out["share_attn"], \
            out["share_lm"], out["share_other"] = \
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    m = RE_DISKCLASS.search(stdout)
    if m:
        out["cold_loads"] = int(m.group(1)); out["cold_gb"] = float(m.group(2))
    return out


def run_case(engine, model, throttle, label, env_extra, cap, ctx, ngen, stats_path):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("IOTH_"):
            del env[k]
    env.update(env_extra)
    env["SNAP"] = model
    env["REF"] = os.path.join(model, "ref_replay.json")
    env["REPLAY"] = "1"
    env["NGEN"] = str(ngen)
    env["CTX"] = str(ctx)
    env["KVSAVE"] = "0"
    env["COLI_USAGE"] = ""           # fresh engine process: no learned history
    env["USAGE_SAVE"] = "0"           # do not write history back (clean A/B)
    if throttle:
        env["LD_PRELOAD"] = throttle
        env["IOTH_MODEL_DIR"] = model
        env["IOTH_LAT_US"] = str(env_extra.get("IOTH_LAT_US", "100"))
        env["IOTH_BW_MBPS"] = str(env_extra.get("IOTH_BW_MBPS", "3500"))
    env["IOTH_STATS"] = stats_path
    cmd = ["/usr/bin/time", "-v", engine, str(cap), "16", "16"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)
        wall, rc = time.time() - t0, p.returncode
        so, se = p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        rc = -9
        so = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        se = "TIMEOUT after 2400s"
    metrics = parse_profile(so)
    m = RE_RSS.search(se)
    if m:
        metrics["peak_rss_mb"] = round(int(m.group(1)) / 1024, 1)
    if throttle and os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                toks = f.read().split()
            metrics["io_nreads"] = int(toks[1]); metrics["io_bytes"] = int(toks[3])
            metrics["io_wait_s"] = round(int(toks[5]) / 1e6, 3)
        except Exception as ex:
            metrics["iothrottle_parse_error"] = str(ex)
    metrics.update({"label": label, "wall_s": round(wall, 2), "rc": rc})
    if rc != 0:
        metrics["stderr_tail"] = se[-800:]
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.expanduser("~/colibri/c/colibri"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--throttle-so", default=None,
                    help="path to iothrottle.so (no I/O emulation if omitted)")
    ap.add_argument("--lat-us", type=float, default=100.0)
    ap.add_argument("--bw-mbps", type=float, default=3500.0)
    ap.add_argument("--profile", choices=["cold", "steady", "longctx", "all"], default="all")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--skip", default="", help="comma-separated labels to skip")
    args = ap.addarser = ap.parse_args()

    model = os.path.abspath(args.model)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    results = {"meta": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": args.engine,
        "model": model,
        "io_emulation": {"lat_us": args.lat_us, "bw_mbps": args.bw_mbps} if args.throttle_so else None,
        "note": "synthetic container (random weights) — no real model downloaded",
    }, "runs": []}

    def cases():
        # (label, env_extra, cap, ctx, ngen)
        # CAP_RAISE=0 everywhere: the default CAP_RAISE=1 silently re-grows the
        # cache cap when RAM allows ("cap raised 64->128"), which made the
        # first pass of this matrix measure the SAME warm operating point for
        # every cap. DIRECT=1 routes expert reads past the page cache so the
        # throttle layer's latency+bandwidth actually apply (else the 2 GB
        # synthetic container lives in RAM and every "miss" is a RAM-speed hit).
        if args.profile in ("cold", "all"):
            yield "cold_cap8_direct", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "0"}, 8, 1024, 512
            yield "cold_cap8_direct_pipe1", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1"}, 8, 1024, 512
            yield "cold_cap8_uring", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "URING": "1"}, 8, 1024, 512
            yield "cold_cap64_direct_pipe1", {"CAP": "64", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1"}, 64, 1024, 512
            yield "cold_cap8_warm_pagecache", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "0", "PIPE": "0"}, 8, 1024, 512
        if args.profile in ("steady", "all"):
            yield "steady_cap8_sata_emul", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1",
                                            "IOTH_BW_MBPS": "500", "IOTH_LAT_US": "150"}, 8, 1024, 512
        if args.profile in ("longctx", "all"):
            yield "longctx_128k_f32kv", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1"}, 8, "131072", 512
            yield "longctx_128k_kv8", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1", "KV8": "1"}, 8, "131072", 512
            yield "longctx_128k_kvtq4", {"CAP": "8", "CAP_RAISE": "0", "DIRECT": "1", "PIPE": "1", "KV_TQ": "4"}, 8, "131072", 512

    did = 0
    for label, env_extra, cap, cts, ngen in cases():
        if label in skip:
            continue
        for r in range(args.runs):
            sp = f"/tmp/ioth_{label}_{r}.txt"
            if os.path.exists(sp):
                os.remove(sp)
            use_throttle = "throttle" in label and args.throttle_so
            m = run_case(args.engine, model, args.throttle_so if use_throttle else None,
                         f"{label}_r{r}", env_extra, cap, cts, ngen, sp)
            results["runs"].append(m)
            did += 1
            key = f"{m.get('tok_s','?')} tok/s" if "tok_s" in m else "FAILED"
            print(f"[{did}] {label}_r{r}: rc={m['rc']} {key} hit={m.get('hit_pct','?')}% "
                  f"rss={m.get('peak_rss_mb','?')}MB", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    print(f"done: {did} runs -> {args.out}")


if __name__ == "__main__":
    main()
