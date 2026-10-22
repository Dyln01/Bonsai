#!/usr/bin/env python3
"""exp_window_sim.py — Phase 4b slice 2: offline window-policy simulation.

    exp_window_sim.py <container_dir> <dump_prefix> <out.json> [--caps 64,128,...]

Reads Q36_PROFILE_TOK dumps (`<prefix>_L<layer>.bin`: per token the hidden row
[D] f32, then K uint32 indices, then K f32 energies — K from Q36_PROFILE_TOK_K)
and replays each layer's per-token active sets through window policies:

  fifo / lru    the WINDOWING_QWEN36.md cache, fed the true per-token sets
  static        cache = token-0 set forever (the degenerate baseline)

plus set-variability stats (mean Jaccard between consecutive tokens, fraction
of tokens whose set differs from token 0's). On synthetic containers the sets
are constant (Phase 4a), so this validates bookkeeping and policy choice; the
curve that decides the window's fate runs on real weights (Phase 5 arm).

No engine, no network: pure replay. rc!=0 if no layer dumps found.
"""
import json
import os
import struct
import sys


def parse_layer(path, D):
    raw = open(path, "rb").read()
    K = None
    rec = None
    # record = D*4 + K*8 bytes; several K can divide the file evenly, so use the
    # dump's invariant: energies are written sorted descending within a record
    for k in (16, 32, 64, 128, 256, 512):
        r = D * 4 + k * 8
        if not raw or len(raw) % r:
            continue
        ce = struct.unpack("<%df" % k, raw[D * 4 + k * 4:D * 4 + k * 8])
        if all(ce[i] >= ce[i + 1] for i in range(k - 1)):
            K, rec = k, r
            break
    if K is None:
        raise SystemExit("%s: no candidate K with sorted energies (D=%d)" % (path, D))
    n = len(raw) // rec
    sets = []
    for i in range(n):
        off = i * rec + D * 4
        ci = struct.unpack("<%dI" % K, raw[off:off + K * 4])
        sets.append(ci)
    return sets, n


def sim(sets, cap):
    """Replay per-token active sets through fifo / lru / static policies."""
    tot = 0
    hits = {"fifo": 0, "lru": 0, "static": 0}
    q = {"fifo": [], "lru": []}
    c = {"fifo": set(), "lru": set()}
    static = set(sets[0]) if sets else set()
    for s in sets:
        for o in s:
            tot += 1
            if o in static:
                hits["static"] += 1
            for pol in ("fifo", "lru"):
                if o in c[pol]:
                    hits[pol] += 1
                    if pol == "lru":
                        q[pol].remove(o); q[pol].append(o)
                else:
                    q[pol].append(o); c[pol].add(o)
                    if len(q[pol]) > cap:
                        c[pol].discard(q[pol].pop(0))
    return {k: (hits[k] / tot if tot else 0.0) for k in hits}


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    caps = [64, 128, 256, 512, 1024, 2048]
    for a in sys.argv[1:]:
        if a.startswith("--caps"):
            caps = [int(x) for x in a.split("=", 1)[1].split(",")]
    if len(args) != 3:
        raise SystemExit(__doc__)
    cont, prefix, out = args
    meta = json.load(open(os.path.join(cont, "qwen36_meta.json")))
    D = meta["hidden"]
    layers = []
    li = 0
    while os.path.exists("%s_L%d.bin" % (prefix, li)):
        layers.append(li); li += 1
    if not layers:
        raise SystemExit("no layer dumps at %s_L*.bin" % prefix)
    result = {"container": cont, "D": D, "layers": [], "caps": caps}
    for li in layers:
        sets, n = parse_layer("%s_L%d.bin" % (prefix, li), D)
        var = {"n_tokens": n,
               "mean_jaccard_consecutive": sum(jaccard(sets[i], sets[i + 1]) for i in range(n - 1)) / max(1, n - 1),
               "frac_differ_from_token0": sum(1 for s in sets if set(s) != set(sets[0])) / n}
        curves = {}
        for c in caps:
            curves[c] = sim(sets, c)
        result["layers"].append({"layer": li, "variability": var, "hitrate_by_cap": curves})
        print("L%-2d tokens=%d jacc(t,t-1)=%.3f differ-from-t0=%.2f | " % (
            li, n, var["mean_jaccard_consecutive"], var["frac_differ_from_token0"]) +
              " ".join("cap%d:fifo%.2f/lru%.2f/stat%.2f" % (c, curves[c]["fifo"], curves[c]["lru"], curves[c]["static"]) for c in caps))
    agg = {}
    for c in caps:
        agg[c] = {pol: sum(l["hitrate_by_cap"][str(c) if str(c) in l["hitrate_by_cap"] else c][pol] for l in result["layers"]) / len(result["layers"])
                  for pol in ("fifo", "lru", "static")}
    result["mean_hitrate_by_cap"] = agg
    json.dump(result, open(out, "w"), indent=1)
    print("mean over layers:", {c: {k: round(v, 3) for k, v in agg[c].items()} for c in caps})
    print("wrote", out)


if __name__ == "__main__":
    main()
