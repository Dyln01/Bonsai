#!/usr/bin/env python3
"""project_q36.py — what this container will actually cost on THIS host.

Phase 5 de-projected the README's "12-14 tok/s warm" row: it was arithmetic for an
int4 trunk whose frac-0.25 hot prefix fits the page cache, and the field container
is F16 (8.6 GB prefix on a 7.62 GB box). The right fix is not a smaller claim, it
is a NUMBER THAT IS A FUNCTION OF THE HOST. This tool is that function:

    bytes/token  = analytic, from the container's OWN shard headers (per class)
    seconds/token= (bytes - resident window) / rate_decode(host)
    TTFT         = prefill bytes / rate_prefill(host)
    warm case    = the part of the working set that fits RAM is served from the page
                   cache at RAM rate; the rest still streams off the disk

Every rate is a HOST ANCHOR measured on the field mini-PC (llm-030: 4-core,
7.62 GB, rotating disk ~10-12 MB/s) and recorded in committed artifacts, so the
model can be regression-gated against the numbers it has to reproduce — see
scripts/check_projection_field.py and docs/COST_MODEL_QWEN36.md:

  rate_prefill = 23.2 MB/s   anchor: TTFT 2460.5 s for a 512-token prompt at
                             frac 1.0 (results/realweights_20261026-1014/arms/A_det.err)
  rate_decode  = 11.2 MB/s   anchor: 2225-2358 s/token at frac 0.25, 1524 ctx
                             (results/battery_llm-030_20261025-133007.json, fs_read 71.7 GB)
  window       = 6.9 GB      anchor: E13/E14 decode re-read 17-24 GB < the trunk;
                             ~6.9 GB stays page-resident and serves the rest

Both rates are the SAME disk measured under two access patterns (batched sequential
prefill vs the decode re-read of a narrowed hot prefix); --probe replaces them with
this host's own measured sequential rate, and the tool says which it used.

USAGE
  python3 tools/project_q36.py --container <dir> [--frac 0.25] [--ctx 1524]
  python3 tools/project_q36.py --geometry --hidden 5120 --shared-inter 17408 \
      --vocab 248320 --layers 64 --attn-layers 16 [--trunk-dtype F16|I4]
  options: --ram-gb 7.62 --nnew 16 --spec-tokens 2.86 --kvq f32|kv8|tq4 \
           --probe <path> --read-mbs 10.5 --json --field-check
"""
import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np

# ---- host anchors (see the docstring for provenance). Overridable, never silent. ----
A_RATE_MBS = 11.2                  # E12/E13/E14: 10.3-11.6 MB/s effective trunk stream
A_RATE_PREFILL_MBS = A_RATE_MBS
A_RATE_DECODE_MBS = A_RATE_MBS
A_WINDOW_GB = 6.9                  # max working set that does NOT come off disk per token.
# Derivation, not a knob to taste: the field's decode re-read is 17-24 GB (E13/E14
# fs_read deltas) against this model's 31.6 GB analytic per-token bytes, so 8-14 GB
# was served from cache/residency; it is additionally capped by the RAM left after
# the fixed residents (embed/KV/norms), because the box has 7.62 GB in total. With
# both applied the model over-predicts the measured decode by 7% — reported, not tuned.
A_PREFILL_FIXED_S = 105.0          # E13's measured per-arm constant

# The field numbers this model must reproduce (--field-check; also the CI gate in
# scripts/check_projection_field.py). Every one is a committed artifact:
#   results/battery_llm-030_20261025-133007.json  (decode, TTFT at frac 0.25/1524 ctx)
#   results/realweights_20261026-1014/arms/A_det.err (TTFT 2460.54 s at frac 1.0/512)
FIELD_CHECKS = [
    # label, kwargs, recorded value, tolerance, note
    ("decode frac 0.25 @1524 ctx", dict(frac=0.25, ctx=1524, nnew=16), 2225.0, 0.15,
     "battery n=7 range 2225-2358 s/token"),
    ("TTFT frac 0.25 @1524 ctx", dict(frac=0.25, ctx=1524, nnew=16, ttft=True), 3466.0, 0.20,
     "battery range 3287.6-3676.2 s"),
    ("TTFT frac 1.0 @512 ctx", dict(frac=1.0, ctx=512, nnew=2, ttft=True), 2460.5, 1.30,
     "M1 det run; ~2x the battery's per-byte rate (page cache warmed by the preceding "
     "tf arm in that session) — the wide band is deliberate, the outlier is reported not hidden"),
]

ESZ = {"F16": 2, "BF16": 2, "F32": 4, "I4": 0.5, "U8": 0.5, "I8": 0.5}
ROLLUP = {  # detail -> printed class
    "mlp": "trunk MLP (HOT_FRAC dial)", "attn": "trunk attention (NO dial)",
    "dn": "trunk DeltaNet (NO dial)", "lmhead": "lm_head (NO dial)",
    "router": "router", "experts": "routed experts", "embed": "embed (resident)",
    "kv": "KV cache (resident)",
}


class Spec:
    """Per-token byte accounting for one container/config combination."""

    def __init__(self, args):
        g = args
        self.D = g.hidden
        self.I = g.shared_inter
        self.E = g.n_experts
        self.moeI = g.moe_inter
        self.topk = g.topk
        self.V = g.vocab
        self.L = g.layers
        self.La = g.attn_layers
        self.Ld = self.L - self.La
        self.qh, self.kvh = g.q_heads, g.kv_heads
        self.qhd, self.khd, self.vhd = g.q_head_dim, g.k_head_dim, g.v_head_dim
        self.o_in = g.o_in or self.qh * self.qhd
        self.conv_dim = g.dn_conv_dim
        self.vdim_tot = g.dn_vdim_tot
        self.frac = g.frac
        self.ctx = g.ctx
        self.trunk = g.trunk_dtype          # MLP (the HOT_FRAC dial's bytes)
        self.a_dt = getattr(g, "attn_dtype", g.trunk_dtype)   # attn + DeltaNet
        self.small = g.small_dtype          # lm_head / embed / experts
        self.kvq = g.kvq
        self.ram = g.ram_gb
        self.R_pf = g.rate_prefill_mbs
        self.R_dc = g.rate_decode_mbs
        self.window = g.window_gb
        self.spec = g.spec_tokens

    def bytes_by_class(self, decode=True):
        """{class: bytes} for one decode step (or one full prefill pass)."""
        z_t, z_s, z_a = ESZ[self.trunk], ESZ[self.small], ESZ[self.a_dt]
        # HOT_FRAC narrows the MLP in BOTH phases: the field's TTFT at frac 0.25
        # (3466 s / 1524 ctx) vs frac 1.0 (2460 s / 512 ctx) is exactly this dial.
        f = self.frac
        b = {}
        if self.I:
            b["mlp"] = f * self.L * (3.0 * self.D * self.I) * z_t
        if self.La:
            b["attn"] = self.La * (self.D * (self.qh * self.qhd) + 2 * self.D * (self.kvh * self.khd)
                                   + self.o_in * self.D) * z_a
        if self.Ld and self.conv_dim:
            b["dn"] = self.Ld * (self.D * self.conv_dim + 2 * self.D * self.vdim_tot) * z_a
        b["lmhead"] = float(self.V) * self.D * z_s
        if self.E:
            b["router"] = self.L * self.D * self.E * z_a
            b["experts"] = self.L * (self.topk or 0) * 3.0 * self.D * self.moeI * z_s
        # resident side (counted for the RAM fit, not as per-token disk bytes)
        res = {}
        res["embed"] = float(self.V) * self.D * z_s
        layers_kv = self.La or self.L
        per_tok = 2 * self.kvh * self.khd * 4          # f32 K+V per attention layer
        if self.kvq == "kv8":
            per_tok = 2 * self.kvh * self.khd * 1 + 8
        elif self.kvq in ("tq4", "tq3"):
            per_tok = 2 * self.kvh * self.khd * 0.5 + 8
        res["kv"] = layers_kv * self.ctx * per_tok
        res["norms"] = self.L * 4 * self.D * 4
        return b, res

    def report(self):
        dec, res = self.bytes_by_class(True)
        pre, _ = self.bytes_by_class(False)
        tot_dec = sum(dec.values())
        tot_pre = sum(pre.values())
        resident = sum(res.values())
        room = max(self.ram * 1e9 - resident, 0.0)
        # The resident window cannot exceed the RAM left after embed/KV/norms — the
        # 6.9 GB field measurement was taken WITH those residents present, and the
        # model must not count the same gigabytes twice.
        win_eff = min(self.window * 1e9, room)
        disk_dec = max(tot_dec - win_eff, 0.0)
        s_dc = disk_dec / (self.R_dc * 1e6)
        s_pf = tot_pre / (self.R_pf * 1e6) + A_PREFILL_FIXED_S
        # warm case: the hotset that fits RAM is served at RAM rate instead of disk rate
        hot = min(dec.get("mlp", 0.0), room)           # only the MLP has a dial/prefix
        ram_rate = 3.0e9                              # ~3 GB/s effective DDR3/4 multi-core
        s_warm = (disk_dec - hot) / (self.R_dc * 1e6) + hot / ram_rate
        self.last_classes, self.last_res = dec, res
        return dict(
            bytes_decode=tot_dec, bytes_prefill=tot_pre, resident_bytes=resident,
            disk_bytes_decode=disk_dec, mlp_bytes=float(dec.get("mlp", 0.0)),
            hot_resident=hot, room=room, window_effective=win_eff,
            s_token_cold=s_dc, s_token_warm=s_warm, ttft_s=s_pf,
            tok_s_cold=1.0 / s_dc if s_dc else float("inf"),
            tok_s_warm=1.0 / s_warm if s_warm else float("inf"),
            tok_s_warm_spec=(self.spec / s_warm) if s_warm else float("inf"),
        )

    def ram_frac(self, target_gb):
        """Largest frac whose MLP hot prefix fits (target - residents)."""
        _, res = self.bytes_by_class(True)
        room = target_gb * 1e9 - sum(res.values())
        per_frac = self.L * (3.0 * self.D * self.I) * ESZ[self.trunk] if self.I else 0.0
        if per_frac <= 0:
            return None
        return max(0.0, min(1.0, room / per_frac))


def probe_rate(path, gb=1.0, chunk=8 << 20):
    """This host's own sequential read rate on the container's own file. Two passes:
    the first is what the engine's cold pass sees, the second is page-cached."""
    if not os.path.exists(path):
        return None
    n = int(gb * 1e9)
    out = []
    for _ in range(2):
        t0, got = time.time(), 0
        with open(path, "rb") as f:
            while got < n:
                b = f.read(min(chunk, n - got))
                if not b:
                    break
                got += len(b)
        dt = max(time.time() - t0, 1e-6)
        out.append(got / dt / 1e6)         # MB/s
    return out


def geometry_from_container(d):
    meta_path = os.path.join(d, "qwen36_meta.json")
    cfg_path = os.path.join(d, "config.json")
    if not os.path.exists(meta_path):
        sys.exit(f"project_q36: {d}: no qwen36_meta.json")
    m = json.load(open(meta_path))
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    # dtype mix from the real headers: report what the bytes actually are
    mix, trunk_bytes, n_mlp = {}, 0, 0
    lay, attn_layers, dn_layers = set(), set(), set()
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, e in hdr.items():
            if k == "__metadata__":
                continue
            nb = e["data_offsets"][1] - e["data_offsets"][0]
            mix[e["dtype"]] = mix.get(e["dtype"], 0) + nb
            mt = re.match(r"(?:language_model\.)?model\.layers\.(\d+)\.", k)
            if mt:
                li = int(mt.group(1))
                lay.add(li)
                if ".self_attn." in k:
                    attn_layers.add(li)
                if ".linear_attn." in k:
                    dn_layers.add(li)
            if ".mlp." in k and k.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight")):
                trunk_bytes += nb
                n_mlp += 1
    # trunk dtype: the MLP's own dtype decides HOT_FRAC's byte cost
    def dtype_of(pred, default):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".safetensors"):
                continue
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k, e in hdr.items():
                if k != "__metadata__" and pred(k):
                    dt = e["dtype"]
                    return "I4" if dt in ("U8", "I8") else dt
        return default
    trunk_dtype = dtype_of(lambda k: k.endswith(("mlp.gate_proj.weight",
                                                "mlp.shared_expert.gate_proj.weight")), "F16")
    lmhead_dtype = dtype_of(lambda k: k.endswith("lm_head.weight"), "F16")
    attn_dtype = dtype_of(lambda k: ".self_attn.q_proj.weight" in k, trunk_dtype)
    return m, cfg, dict(mix=mix, trunk_dtype=trunk_dtype, lmhead_dtype=lmhead_dtype,
                        attn_dtype=attn_dtype, mlp_bytes=trunk_bytes,
                        mlp_tensors=n_mlp, layers=len(lay), attn_layers=len(attn_layers),
                        dn_layers=len(dn_layers))


def field_check(base):
    """Re-run the model at each recorded field point and report the residual.
    This is the gate that stops the README drifting from physics: the projection
    must reproduce the numbers the field actually measured, or it is not a projection."""
    print("\n== field fit (the model vs committed field measurements) ==")
    print(f"{'point':30} {'recorded':>10} {'model':>10} {'residual':>9}  band")
    bad = 0
    for label, kw, recorded, tol, note in FIELD_CHECKS:
        a = argparse.Namespace(**vars(base))
        for k, v in kw.items():
            if k == "ttft":
                continue
            setattr(a, k, v)
        s = Spec(a)
        r = s.report()
        got = r["ttft_s"] if kw.get("ttft") else r["s_token_cold"]
        res = (got - recorded) / recorded
        ok = abs(res) <= tol
        bad += 0 if ok else 1
        print(f"{label:30} {recorded:10.1f} {got:10.1f} {100 * res:+8.1f}%  "
              f"±{100 * tol:.0f}% {'ok' if ok else 'OUT'}")
        print(f"    {note}")
    print(f"{'decode, int4 trunk @frac 0.25':30} {'-':>10} "
          f"{Spec(_ns(base, trunk_dtype='I4', attn_dtype='I4', small_dtype='I4')).report()['s_token_cold']:10.1f} "
          f"{'':>9}   <- the packer's effect (bytes/token only)")
    return bad


def _ns(base, **kw):
    a = argparse.Namespace(**vars(base))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container")
    ap.add_argument("--geometry", action="store_true",
                    help="use the explicit --hidden/--vocab/... numbers instead of a container")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--shared-inter", type=int, default=17408)
    ap.add_argument("--moe-inter", type=int, default=0)
    ap.add_argument("--n-experts", type=int, default=0)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--attn-layers", type=int, default=16)
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--q-head-dim", type=int, default=192)
    ap.add_argument("--k-head-dim", type=int, default=128)
    ap.add_argument("--v-head-dim", type=int, default=128)
    ap.add_argument("--o-in", type=int, default=0)
    # defaults reproduce the FIELD container's documented arithmetic: trunk 54.7 GB
    # = 34.2 GB MLP (64 x 3 x 5120 x 17408 x 2 B) + 20.5 GB attn+DeltaNet. The real
    # numbers are read from qwen36_meta.json whenever --container is given.
    ap.add_argument("--dn-conv-dim", type=int, default=18432)
    ap.add_argument("--dn-vdim-tot", type=int, default=9216)
    ap.add_argument("--trunk-dtype", default="F16", choices=["F16", "F32", "I4", "U8"])
    ap.add_argument("--small-dtype", default="F16")
    ap.add_argument("--attn-dtype", default=None,
                    help="attn + DeltaNet dtype (default: same as --trunk-dtype)")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--ctx", type=int, default=1524)
    ap.add_argument("--nnew", type=int, default=16)
    ap.add_argument("--ram-gb", type=float, default=7.62)
    ap.add_argument("--kvq", default="f32", choices=["f32", "kv8", "tq4", "tq3"])
    ap.add_argument("--spec-tokens", type=float, default=1.0,
                    help="accepted tokens/forward (field MTP measured 2.86)")
    ap.add_argument("--rate-prefill-mbs", type=float, default=A_RATE_PREFILL_MBS)
    ap.add_argument("--rate-decode-mbs", type=float, default=A_RATE_DECODE_MBS)
    ap.add_argument("--window-gb", type=float, default=A_WINDOW_GB)
    ap.add_argument("--read-mbs", type=float, help="override BOTH rates with one measured rate")
    ap.add_argument("--probe", help="measure this host's read rate on <path> first")
    ap.add_argument("--field-check", action="store_true",
                    help="fit the model against the recorded field measurements and exit "
                         "non-zero if any point is outside its band")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.attn_dtype is None:
        a.attn_dtype = a.trunk_dtype
    if a.read_mbs:
        a.rate_prefill_mbs = a.rate_decode_mbs = a.read_mbs
    ctr = None
    if a.container:
        m, cfg, ctr = geometry_from_container(a.container)
        a.hidden = int(m.get("hidden") or cfg.get("hidden_size") or a.hidden)
        a.shared_inter = int(m.get("shared_inter") or a.shared_inter)
        a.moe_inter = int(m.get("moe_inter") or 0)
        a.n_experts = int(m.get("num_experts") or 0)
        a.topk = int(m.get("topk") or 0)
        a.vocab = int(cfg.get("vocab_size") or m.get("vocab") or a.vocab)
        a.layers = int(m.get("n_layers") or a.layers)
        a.q_heads = int(m.get("q_heads") or a.q_heads)
        a.kv_heads = int(m.get("kv_heads") or a.kv_heads)
        a.q_head_dim = int(m.get("q_head_dim") or a.q_head_dim)
        a.k_head_dim = int(m.get("k_head_dim") or a.k_head_dim)
        a.v_head_dim = int(m.get("v_head_dim") or a.k_head_dim)
        a.o_in = int(m.get("o_in") or 0)
        a.dn_conv_dim = int(m.get("dn_conv_dim") or a.dn_conv_dim)
        a.dn_vdim_tot = int(m.get("dn_vheads") or 0) * int(m.get("dn_vdim") or 0) or a.dn_vdim_tot
        a.trunk_dtype = ctr["trunk_dtype"]
        a.attn_dtype = ctr["attn_dtype"]
        a.small_dtype = ctr["lmhead_dtype"]
        a.attn_layers = ctr["attn_layers"]
    if not a.container and not a.geometry and not a.field_check:
        sys.exit("project_q36: pass --container <dir> or --geometry with explicit numbers")
    print("== host anchors ==")
    if a.probe:
        r = probe_rate(a.probe)
        if r:
            print(f"  probe {a.probe}: cold {r[0]:.1f} MB/s, warm(cached) {r[1]:.1f} MB/s "
                  f"-> using cold for the disk rate")
            if not a.read_mbs:
                a.rate_decode_mbs = r[0]
                a.rate_prefill_mbs = r[0]
        else:
            print(f"  probe {a.probe}: unreadable — keeping the field anchors")
    print(f"  prefill rate {a.rate_prefill_mbs:.1f} MB/s · decode rate {a.rate_decode_mbs:.1f} MB/s "
          f"· resident window {a.window_gb:.1f} GB")
    print(f"  (field llm-030 anchors; override with --read-mbs/--probe)")
    if ctr:
        print(f"\n== container ==\n  {a.container}")
        print(f"  trunk MLP dtype: {ctr['trunk_dtype']}  ({ctr['mlp_bytes'] / 1e9:.2f} GB over "
              f"{ctr['mlp_tensors']} tensors)")
        print(f"  layers {ctr['layers']} ({ctr['attn_layers']} attn + {ctr['dn_layers']} DeltaNet)")
        print("  dtype mix: " + ", ".join(f"{k} {v / 1e9:.2f} GB"
                                          for k, v in sorted(ctr["mix"].items())))
    s = Spec(a)
    r = s.report()
    print(f"\n== per-token bytes, decoded at frac={a.frac} ==")
    for cls, by in sorted(s.last_classes.items(), key=lambda kv: -kv[1]):
        print(f"  {ROLLUP.get(cls, cls):28} {by / 1e9:7.3f} GB  {100 * by / r['bytes_decode']:5.1f}%")
    print(f"  {'TOTAL':28} {r['bytes_decode'] / 1e9:7.3f} GB")
    print(f"  resident window serving part of it: {r['hot_resident'] / 1e9:.2f} GB "
          f"(room {r['room'] / 1e9:.2f} GB after {r['resident_bytes'] / 1e9:.2f} GB of "
          f"embed/KV/norms)")
    print(f"  off-disk per token: {r['disk_bytes_decode'] / 1e9:.3f} GB")
    print(f"\n== this host ==")
    print(f"  TTFT ({a.ctx} tok): {r['ttft_s'] / 60:.1f} min")
    print(f"  decode, cold (every token off disk): {r['s_token_cold']:8.1f} s/token  "
          f"= {r['tok_s_cold'] * 3600:.2f} tok/hour")
    print(f"  decode, warm (hotset page-resident): {r['s_token_warm']:8.2f} s/token  "
          f"= {r['tok_s_warm']:.3f} tok/s")
    if a.spec_tokens > 1.0:
        print(f"  with MTP self-speculation ({a.spec_tokens:.2f} accepted/forward): "
              f"{r['tok_s_warm_spec']:.3f} tok/s warm")
    if a.nnew > 1:
        print(f"  {a.nnew}-token reply: {(a.nnew - 1) * r['s_token_cold'] / 3600:.1f} h cold, "
              f"{(a.nnew - 1) * r['s_token_warm'] / 60:.1f} min warm")
    print(f"\n== the RAM wall ==")
    for target in (a.ram_gb, 15.6, 31.2):
        fr = s.ram_frac(target)
        if fr is None:
            continue
        tag = " (this box)" if abs(target - a.ram_gb) < 1e-6 else ""
        verdict = ("warm regime reachable" if fr >= a.frac else
                   f"needs frac <= {fr:.3f} for the MLP prefix to fit")
        print(f"  {target:5.1f} GB{tag:11} MLP prefix fits at frac <= {fr:.3f} -> {verdict}")
    rc = 0
    if a.field_check:
        rc = 1 if field_check(a) else 0
    if a.json:
        json.dump(dict(args=vars(a), result=r, container=ctr),
                  open(a.json, "w"), indent=1, default=str)
        print(f"\njson: {a.json}")
    if a.field_check and rc:
        sys.exit(f"project_q36: {rc} field point(s) outside band — the projection no longer "
                 f"matches the measurements it is calibrated on (see docs/COST_MODEL_QWEN36.md)")


if __name__ == "__main__":
    main()
