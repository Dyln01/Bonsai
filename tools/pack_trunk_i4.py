#!/usr/bin/env python3
"""pack_trunk_i4.py — offline dense-trunk int4 packer (the E13 structural follow-up).

WHY. The field container is NAMED q27_i4 but its dense trunk is F16: the pinned
converter quantizes routed experts only and a dense 27B has none. Field decode is
pure weight streaming (compute cores <1% of a token), ~2000-2358 s/token at
~10 MB/s, of which the F16 trunk is nearly all of it. Packing the trunk to int4
is the ONLY structural lever on that floor: 4x fewer bytes per token, and it
unblocks three things stuck on one wall — the per-layer window cache
(`q36_win_ensure` requires map_len == D*I/2), the sorted-container arm E (sort
output 50 GB -> ~14 GB), and honest per-class byte accounting.

THE DIALECT (engine qwen36.c, load_tq -> matmul_map_i4 / q36_down_prefix):
  * a mapped int4 dense tensor is `dtype == 3` (safetensors "U8") with
    `nbytes == I*O/2`; byte k holds element 2k in the LOW nibble and element
    2k+1 in the HIGH nibble, two's-complement signed (matmul_map_i4).
  * its scale sidecar is `<name>.qs`: "F32", numel == O (one scale per output
    row); the engine keeps it resident. No .qs -> scales read as 1.0.
  * quantizer (mirrors scripts/gen_student_i4.py quant_i4_row, the layout every
    CI gate and synthetic container already uses):
        amax = max|row|;  s = max(amax/7, 1e-8);  q = rint(clip(w*7/max(amax,1e-8), -7, 7))
    so dequant is q*s and the engine's dot reads the row exactly as packed.
  * layout is row-major [O][I] flat 1-D (the v2 contract hot_sort.py writes);
    HOT_FRAC narrows to the first Ik rows/cols, so the sort order IS the file
    order — pass --profile to write the hotness-sorted prefix in the same pass.

WHAT IT TOUCHES. Only the dense GEMV tensors the engine reads through
load_tq(quantize=1): lm_head, the MLP (dense `mlp.{gate,up,down}_proj` or MoE
`mlp.shared_expert.*`), the MoE router, the attention q/k/v/o projections and
the DeltaNet in_proj_qkv / in_proj_z / out_proj. Every other tensor (norms,
embed, conv1d, b/a, dt_bias, A_log, experts) is copied through verbatim.

SAFETY: refuses to run twice on the same tensor (already U8/I8 -> hard error, never
a silent double-pack), refuses geometry that disagrees with qwen36_meta.json,
mirrors st_init's duplicate-shard rule, and `--verify` re-reads the output and
proves the engine's read predicate plus dequant parity per class.

USAGE
  python3 tools/pack_trunk_i4.py <src_dir> <out_dir> [options]
    --components a,b,c   subset of lmhead,shexp,attn,dnproj,dnout,router (default all)
    --profile <json>     hot_profile.json -> MLP gate/up rows + down cols permuted
                         by argsort(-energy), the hot prefix first (hot_sort rule)
    --block-rows N       streaming row block (default 256)
    --dry-run            report the target set + byte budget; write nothing
    --verify             after writing: predicates + per-class dequant parity
    --emit-dequant <dir> write a second container whose targets are the DEQUANTIZED
                         values as F16 — the oracle a CI parity gate runs against
    --hash               also sha256 every shard in and out (slow: full re-read)
    --force              overwrite a non-empty out_dir (default: refuse)

Output: <out_dir>/ = all non-safetensors files copied, every shard rewritten with
packed targets + .qs sidecars in place, plus trunk_i4_manifest.json.
No engine change is needed: the engine dispatches on dtype/nbytes alone.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

ESZ = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F64": 8, "I64": 8, "I32": 4, "U32": 4}
NP = {"F16": np.float16, "F32": np.float32}
FLOAT_IN = ("F16", "F32")          # what this packer can read (BF16: refuse, see load_block)
QS_SUFFIX = ".qs"

def die(msg):
    print(f"pack_trunk_i4: error: {msg}", file=sys.stderr)
    sys.exit(1)

def note(msg):
    print(f"pack_trunk_i4: {msg}", file=sys.stderr)

def read_header(path):
    """safetensors header -> {name: {dtype, shape, data_offsets}} + data start."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            die(f"{path}: not a safetensors file (truncated length prefix)")
        n = struct.unpack("<Q", raw)[0]
        hj = f.read(n)
        if len(hj) != n:
            die(f"{path}: truncated header ({len(hj)} of {n} bytes)")
        hdr = {}
        try:
            hdr = json.loads(hj)
        except Exception as e:
            die(f"{path}: header JSON unreadable: {e}")
    return hdr, 8 + n

# ---- name classification: role + (I, O) from qwen36_meta.json / config.json ----
# Ordered: most specific first. Each pattern is matched against the name with any
# `language_model.` prefix stripped (dense_resolve tries plain, then prefixed).
ROLE_PATTERNS = [
    (re.compile(r"^lm_head\.weight$"),                                   "lmhead"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"),           "router"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj\.weight$"), "sh_gate"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.up_proj\.weight$"),   "sh_up"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.down_proj\.weight$"), "sh_down"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),      "sh_gate"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"),        "sh_up"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),      "sh_down"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"),   "q"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$"),   "k"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$"),   "v"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),   "o"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$"), "dn_qkv"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$"),   "dn_z"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$"),    "dn_out"),
]
ROLE_CLASS = {"lmhead": "lmhead", "router": "router",
              "sh_gate": "shexp", "sh_up": "shexp", "sh_down": "shexp",
              "q": "attn", "k": "attn", "v": "attn", "o": "attn",
              "dn_qkv": "dnproj", "dn_z": "dnproj", "dn_out": "dnout"}
# roles whose rows are hotness-permutable (hot_sort's scope: the MLP, I-indexed)
SORTABLE = {"sh_gate": "rows", "sh_up": "rows", "sh_down": "cols"}
class Geo:
    """The (I, O) every mapped dense tensor must match, from qwen36_meta.json +
    config.json — the same numbers load_tq_noted is called with. A container whose
    tensor geometry disagrees is a refusal here, not a silent wrong-size map."""

    def __init__(self, src):
        mp = os.path.join(src, "qwen36_meta.json")
        if not os.path.exists(mp):
            die(f"{src}: no qwen36_meta.json — not a colibri container directory "
                f"(the engine needs it too: run: python3 tools/st_dump.py {src})")
        self.meta = json.load(open(mp))
        cfg = {}
        cp = os.path.join(src, "config.json")
        if os.path.exists(cp):
            try:
                cfg = json.load(open(cp))
            except Exception:
                cfg = {}
        m = self.meta
        self.hidden = int(m.get("hidden") or cfg.get("hidden_size") or 0)
        if self.hidden <= 0:
            die(f"{src}: qwen36_meta.json has no usable 'hidden'")
        self.shared_inter = int(m.get("shared_inter") or 0)
        self.moe_inter = int(m.get("moe_inter") or 0)
        self.n_experts = int(m.get("num_experts") or 0)
        self.vocab = int(cfg.get("vocab_size") or m.get("vocab") or 0)
        self.q_heads = int(m.get("q_heads") or 0)
        self.kv_heads = int(m.get("kv_heads") or 0)
        self.head_dim = int(m.get("head_dim") or 0)
        self.q_head_dim = int(m.get("q_head_dim") or self.head_dim)
        self.k_head_dim = int(m.get("k_head_dim") or self.head_dim)
        self.o_in = int(m.get("o_in") or (self.q_heads * self.head_dim))
        vh = int(m.get("dn_vheads") or 0)
        vd = int(m.get("dn_vdim") or 0)
        self.dn_vdim_tot = vh * vd
        self.dn_conv_dim = int(m.get("dn_conv_dim") or 0)
        if self.dn_conv_dim <= 0:
            kh, kd = int(m.get("dn_kheads") or 0), int(m.get("dn_kdim") or 0)
            self.dn_conv_dim = 2 * kh * kd + vh * vd
        self.mlp_inter = self.shared_inter or self.moe_inter

    def io_for(self, role):
        """(I, O) for a role, or (None, None) when the meta cannot pin it down."""
        D = self.hidden
        if role == "lmhead":
            return (D, self.vocab or None)
        if role == "router":
            return (D, self.n_experts or None)
        if role == "sh_gate" or role == "sh_up":
            return (D, self.mlp_inter or None)
        if role == "sh_down":
            return (self.mlp_inter or None, D)
        if role == "q":
            return (D, (self.q_heads * self.q_head_dim) or None)
        if role == "k" or role == "v":
            return (D, (self.kv_heads * self.k_head_dim) or None)
        if role == "o":
            return (self.o_in or None, D)
        if role == "dn_qkv":
            return (D, self.dn_conv_dim or None)
        if role == "dn_z":
            return (D, self.dn_vdim_tot or None)
        if role == "dn_out":
            return (self.dn_vdim_tot or None, D)
        return (None, None)


def classify(name):
    """-> (role, class, layer_index) for a dense-GEMV tensor name, else None."""
    n = name
    if n.startswith("language_model."):
        n = n[len("language_model."):]
    for rx, role in ROLE_PATTERNS:
        mt = rx.match(n)
        if mt:
            li = int(mt.group(1)) if mt.groups() else -1
            return (role, ROLE_CLASS[role], li)
    return None


def qs_name(name):
    return name + QS_SUFFIX


def build_index(src):
    """name -> (shard_file, header_entry). Mirrors st_init's duplicate rule via
    model.safetensors.index.json (the same contract hot_sort.build_owner_map enforces)."""
    files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not files:
        die(f"{src}: no *.safetensors shards")
    idx = os.path.join(src, "model.safetensors.index.json")
    weight_map = {}
    if os.path.exists(idx):
        try:
            weight_map = (json.load(open(idx)).get("weight_map") or {})
        except Exception as e:
            die(f"{idx}: unreadable index ({e}) — st_init refuses this container")
    headers, owner = {}, {}
    for f in files:
        hdr, dstart = read_header(os.path.join(src, f))
        headers[f] = (hdr, dstart)
        for name in hdr:
            if name == "__metadata__":
                continue
            prev = owner.get(name)
            if prev is None or prev == f:
                owner[name] = f
                continue
            auth = weight_map.get(name)
            if auth == f:
                owner[name] = f
            elif auth == prev:
                pass
            else:
                why = (f"the index maps it to '{auth}', a third file" if auth else
                       ("the index does not list it" if weight_map else
                        "there is no model.safetensors.index.json"))
                die(f"tensor '{name}' is in both '{prev}' and '{f}' and {why} — "
                    f"st_init refuses to load a directory like this; remove the stray "
                    f"copy (run: python3 tools/st_dump.py {src})")
    return files, headers, owner


def plan_targets(geo, components, files, headers, owner, dstart_of=None):
    """-> (targets, skipped) where targets maps name -> dict(role, class, I, O,
    dtype, nbytes, shard, entry) for every tensor this packer will rewrite."""
    targets, skipped = {}, []
    for f in files:
        hdr, _ = headers[f]
        for name, e in hdr.items():
            if name == "__metadata__":
                continue
            if owner.get(name, f) != f:
                continue                      # ignored duplicate: engine reads the other shard
            if name.endswith(QS_SUFFIX):
                continue                      # sidecar; handled with its tensor
            c = classify(name)
            if not c:
                continue
            role, cls, li = c
            dt = e.get("dtype")
            shape = e.get("shape") or []
            numel = int(np.prod(shape)) if shape else 0
            if cls not in components:
                skipped.append((name, f"class '{cls}' not in --components"))
                continue
            I, O = geo.io_for(role)
            if dt in ("U8", "I8"):
                die(f"{f}: '{name}' is already {dt}-packed — refusing to pack twice. "
                    f"This container already has an int4 trunk (or a partially packed "
                    f"one); serve it directly, or start from the F16 container.")
            if dt not in FLOAT_IN:
                die(f"{f}: '{name}': dtype {dt} is not {FLOAT_IN} — the pinned converter's "
                    f"dense trunk is F16 (BF16/F8 dense has no mapped read path in the "
                    f"engine; see docs/TRUNKMAP_DENSE27B.md)")
            if numel % 2:
                die(f"{f}: '{name}': {numel} elements is odd — int4 needs pairs")
            if I and O and numel != I * O:
                die(f"{f}: '{name}': {dt} has {numel:,} elements; qwen36_meta geometry "
                    f"implies {I * O:,} ({O} x {I}) for role '{role}'. The engine's "
                    f"load_t_n refuses this too (numel != I*O), so the shard was not "
                    f"written by the pinned convert_qwen36.py dense path. Run: "
                    f"python3 tools/st_dump.py . and share the output.")
            if not (I and O):
                note(f"{name}: meta could not pin (I,O); packing on the header shape "
                     f"{shape} (the engine will refuse a wrong size at load)")
                I = O = None
            off = e["data_offsets"][0]
            targets[name] = dict(role=role, cls=cls, layer=li, I=I, O=O, dtype=dt,
                                 numel=numel, nbytes=e["data_offsets"][1] - off,
                                 abs_off=(dstart_of or {}).get(f, 0) + off,
                                 shard=f, entry=e)
    return targets, skipped
def quant_rows(w2d):
    """[n, I] float32 -> (packed U8 flat [n*I/2], scales F32 [n]).
    Byte k = element 2k low nibble, 2k+1 high nibble (the converter convention
    matmul_map_i4 reads); scale = max(amax/7, 1e-8) so dequant is code*scale."""
    amax = np.abs(w2d).max(axis=1, keepdims=True)
    safe = np.maximum(amax, 1e-8)
    s = np.maximum(amax / 7.0, 1e-8).astype(np.float32).reshape(-1)
    q = np.rint(np.clip(w2d / safe * 7.0, -7, 7)).astype(np.int8).reshape(-1)
    lo = (q[0::2] & 0xF).astype(np.uint8)
    hi = ((q[1::2] & 0xF) << 4).astype(np.uint8)
    return (lo | hi), s


def load_profile(path, geo):
    """hot_profile.json -> {layer_index: order} using hot_sort's rule
    (argsort(-energy, stable), width must equal the MLP width)."""
    prof = json.load(open(path))
    orders, skipped = {}, 0
    for k, v in prof.items():
        mt = re.fullmatch(r"L(\d+)", k or "")
        if not mt or not isinstance(v, dict) or "energy" not in v:
            continue
        e = np.array(v["energy"], dtype=np.float64)
        if geo.mlp_inter and len(e) != geo.mlp_inter:
            note(f"profile {k}: width {len(e)} != mlp width {geo.mlp_inter} — "
                 f"layer skipped (profile from a different geometry)")
            skipped += 1
            continue
        orders[int(mt.group(1))] = np.argsort(-e, kind="stable")
    if not orders:
        die(f"{path}: no usable L<n> energy entries — calibrate first "
            f"(q36-minipc calibrate), or drop --profile")
    if skipped:
        note(f"profile: {skipped} layer(s) skipped on width mismatch")
    return orders


def read_rows(f, off, rows, I, esz):
    """Read `rows` rows of width I at `off` -> float32 [rows, I].
    Sequential by design: the source may be a 54.7 GB container on a 7.6 GB box."""
    f.seek(off)
    buf = f.read(rows * I * esz)
    want = rows * I * esz
    if len(buf) != want:
        die(f"short read: wanted {want} bytes at {off}, got {len(buf)}")
    if esz == 2:
        return np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(rows, I)
    return np.frombuffer(buf, dtype=np.float32).reshape(rows, I)


def write_header(f, entries, metadata=None):
    hdr = {}
    if metadata is not None:
        hdr["__metadata__"] = metadata
    off = 0
    for name, (dt, nbytes, shape) in entries:
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    return off


def copy_span(fout, fin, off, nbytes, chunk=8 << 20):
    fin.seek(off)
    left = nbytes
    while left > 0:
        b = fin.read(min(chunk, left))
        if not b:
            die(f"short read while copying {nbytes} bytes at {off}")
        fout.write(b)
        left -= len(b)


def shard_plan(hdr, targets_in_shard, geo, dstart=0):
    """Output entries for one shard in source order: packed targets + their
    regenerated .qs, every other tensor (and stray non-target .qs) verbatim."""
    entries, work = [], []
    for name, e in hdr.items():
        if name == "__metadata__":
            continue
        dt = e.get("dtype")
        nbytes = e["data_offsets"][1] - e["data_offsets"][0]
        t = targets_in_shard.get(name)
        if t:
            I, O = t["I"], t["O"]
            entries.append((name, ("U8", I * O // 2, [I * O // 2])))
            entries.append((qs_name(name), ("F32", O * 4, [O])))
            work.append(("pack", name, t, None))
            continue
        if name.endswith(QS_SUFFIX):
            base = name[:-len(QS_SUFFIX)]
            if base in targets_in_shard:
                continue                      # sidecar regenerated from the source floats
        entries.append((name, (dt, nbytes, list(e.get("shape") or []))))
        work.append(("copy", name, None, (dstart + e["data_offsets"][0], nbytes)))
    return entries, work
def deq_plan(hdr, targets_in_shard, dstart=0):
    """Same walk as shard_plan but targets are written DEQUANTIZED as F16 (the
    oracle container: the engine's mapped-F16 read of exactly the values the
    mapped-int4 read reconstructs, so a parity gate isolates nibble order and
    scale placement from all other numerics)."""
    entries, kinds = [], []
    for name, e in hdr.items():
        if name == "__metadata__":
            continue
        dt = e.get("dtype")
        nbytes = e["data_offsets"][1] - e["data_offsets"][0]
        t = targets_in_shard.get(name)
        if t:
            I, O = t["I"], t["O"]
            entries.append((name, ("F16", I * O * 2, [O, I])))
            kinds.append("deq")
            continue
        if name.endswith(QS_SUFFIX) and name[:-len(QS_SUFFIX)] in targets_in_shard:
            continue
        entries.append((name, (dt, nbytes, list(e.get("shape") or []))))
        kinds.append(("copy", dstart + e["data_offsets"][0], nbytes))
    return entries, kinds


def pack_shard(src, fname, hdr, ts, geo, args, orders, deq_dir=None):
    """One streaming pass over one shard: read source floats block-wise, write the
    packed U8 + regenerated F32 .qs, and (optionally) the dequantized F16 oracle.
    Returns (stats, per_shard_sizes)."""
    fin = open(os.path.join(src, fname), "rb")
    _, dstart = read_header(os.path.join(src, fname))
    entries, work = shard_plan(hdr, ts, geo, dstart)
    deq_entries = deq_kinds = None
    if deq_dir:
        deq_entries, deq_kinds = deq_plan(hdr, ts, dstart)
        if len(deq_kinds) != len(work):
            die("internal: dequant plan and pack plan disagree on tensor count")
    out_path = os.path.join(args.out, fname)
    fout = open(out_path, "wb")
    total_main = write_header(fout, entries)
    fdeq = None
    if deq_dir:
        fdeq = open(os.path.join(deq_dir, fname), "wb")
        write_header(fdeq, deq_entries)
    stats = []
    for i, (kind, name, t, span) in enumerate(work):
        if kind == "copy":
            copy_span(fout, fin, span[0], span[1])
            if fdeq:
                assert deq_kinds is not None and deq_kinds[i][0] == "copy" \
                    and deq_kinds[i][1] == span[0]
                copy_span(fdeq, fin, span[0], span[1])
            continue
        I, O, esz = t["I"], t["O"], ESZ[t["dtype"]]
        assert deq_kinds is None or deq_kinds[i] == "deq"
        off = t["abs_off"]
        order = orders.get(t["layer"]) if (t["cls"] == "shexp" and orders) else None
        how = SORTABLE.get(t["role"]) if order is not None else None
        scales = np.empty(O, dtype=np.float32)
        max_abs = 0.0
        sumsq_err = 0.0
        sumsq_ref = 0.0
        done = 0
        while done < O:
            rows = min(args.block_rows, O - done)
            blk = read_rows(fin, off + done * I * esz, rows, I, esz)
            if how == "rows":
                # output row (done+k) must contain source row order[done+k]
                idx = order[done:done + rows]
                blk = np.empty((rows, I), dtype=np.float32)
                for k, sr in enumerate(idx):
                    blk[k] = read_rows(fin, off + int(sr) * I * esz, 1, I, esz)[0]
            elif how == "cols":
                blk = blk[:, order]            # columns dimension == mlp_inter, this one is correct
            packed, s = quant_rows(blk)
            fout.write(packed.tobytes())
            scales[done:done + rows] = s
            if fdeq:
                q = np.frombuffer(packed.tobytes(), dtype=np.uint8).astype(np.int16)
                lo, hi = (q & 0xF), (q >> 4)
                q = np.empty(q.size * 2, dtype=np.int16)
                q[0::2] = np.where(lo & 8, lo - 16, lo)
                q[1::2] = np.where(hi & 8, hi - 16, hi)
                deq = (q.astype(np.float32).reshape(rows, I) * s.reshape(rows, 1))
                fdeq.write(deq.astype(np.float16).tobytes())
                d = deq - blk                        # vs the (possibly permuted) source
                max_abs = max(max_abs, float(np.abs(d).max(initial=0.0)))
                sumsq_err += float((d.astype(np.float64) ** 2).sum())
                sumsq_ref += float((blk.astype(np.float64) ** 2).sum())
            done += rows
        fout.write(scales.tobytes())
        stats.append(dict(name=name, cls=t["cls"], role=t["role"], layer=t["layer"],
                          I=I, O=O, src_dtype=t["dtype"], src_bytes=t["nbytes"],
                          out_bytes=I * O // 2 + O * 4, permuted=bool(how),
                          max_abs_err=max_abs,
                          rel_l2_err=(sumsq_err / sumsq_ref) ** 0.5 if sumsq_ref else 0.0))
        print(f"  {name}: {t['dtype']} {O}x{I} -> U8 {I * O // 2 / 1e6:.1f} MB "
              f"+ qs {O * 4 / 1e6:.2f} MB (rel L2 {stats[-1]['rel_l2_err']:.2e})",
              flush=True)
    fout.close()
    if fdeq:
        fdeq.close()
    fin.close()
    return stats, total_main
def verify_container(out, src, targets, geo, block_bytes=64 << 20):
    """Prove the engine's read predicate on the OUTPUT container (this is the gate
    that matters: load_tq maps a tensor iff dtype=='U8' and nbytes == I*O/2, and
    reads scales iff a 'F32' sidecar with numel == O exists)."""
    fails, checks = [], 0
    files, headers, _ = build_index(out)
    idx = {}
    for f in files:
        dst = headers[f][1]
        for name, e in headers[f][0].items():
            if name != "__metadata__":
                idx[name] = (f, e, dst)
    for name, t in targets.items():
        I, O = t["I"], t["O"]
        if name not in idx:
            fails.append(f"{name}: MISSING from output")
            continue
        f, e, _dst = idx[name]
        checks += 1
        if e.get("dtype") != "U8":
            fails.append(f"{name}: dtype {e.get('dtype')} (engine needs U8 -> dtype code 3)")
        nb = e["data_offsets"][1] - e["data_offsets"][0]
        if nb != I * O // 2:
            fails.append(f"{name}: nbytes {nb} != I*O/2 {I * O // 2} (mapped-int4 predicate)")
        qs = qs_name(name)
        if qs not in idx:
            fails.append(f"{qs}: MISSING (engine reads per-row scales from it)")
        else:
            qe = idx[qs][1]
            qn = qe["data_offsets"][1] - qe["data_offsets"][0]
            if qe.get("dtype") != "F32" or qn != O * 4:
                fails.append(f"{qs}: {qe.get('dtype')} nbytes {qn}; engine needs F32 with "
                             f"numel == O ({O})")
    # passthrough integrity: names present in src and out, not targets, must match
    # byte-for-byte (full compare; tensors over block_bytes are size-checked only —
    # say what was checked, never imply more)
    src_files, src_headers, _ = build_index(src)
    src_idx = {}
    for f in src_files:
        dst = src_headers[f][1]
        for name, e in src_headers[f][0].items():
            if name != "__metadata__":
                src_idx[name] = (os.path.join(src, f), e, dst)
    same = 0
    for name, (path, e, sdst) in sorted(src_idx.items()):
        if name in targets or (name.endswith(QS_SUFFIX) and name[:-len(QS_SUFFIX)] in targets):
            continue
        if name not in idx:
            fails.append(f"{name}: passthrough tensor missing from output")
            continue
        of, oe, odst = idx[name]
        out_path = os.path.join(out, of)
        n = e["data_offsets"][1] - e["data_offsets"][0]
        if oe.get("dtype") != e.get("dtype") or n != (oe["data_offsets"][1] - oe["data_offsets"][0]):
            fails.append(f"{name}: passthrough header changed ({e.get('dtype')} -> {oe.get('dtype')})")
            continue
        checks += 1
        if n > block_bytes:
            continue                      # larger than the verify budget: size-checked only
        a = _read_span(path, sdst + e["data_offsets"][0], n)
        b = _read_span(out_path, odst + oe["data_offsets"][0], n)
        if a == b:
            same += 1
        else:
            fails.append(f"{name}: passthrough bytes differ from source")
    return checks, same, fails


def _read_span(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def dry_run(geo, targets, skipped, files, headers):
    per = {}
    for name, t in targets.items():
        d = per.setdefault(t["cls"], dict(n=0, src=0, out=0))
        d["n"] += 1
        d["src"] += t["nbytes"]
        d["out"] += t["I"] * t["O"] // 2 + t["O"] * 4
    src_trunk = sum(t["nbytes"] for t in targets.values())
    out_trunk = sum(t["I"] * t["O"] // 2 + t["O"] * 4 for t in targets.values())
    print(f"\n=== dry run: what this packer would rewrite ({len(targets)} tensors) ===")
    print(f"{'class':8} {'tensors':>7} {'now':>10} {'after':>10}  ratio")
    for cls in sorted(per):
        d = per[cls]
        print(f"{cls:8} {d['n']:7d} {d['src'] / 1e9:9.2f}G {d['out'] / 1e9:9.2f}G  "
              f"{d['src'] / max(d['out'], 1):5.2f}x")
    print(f"{'TOTAL':8} {len(targets):7d} {src_trunk / 1e9:9.2f}G {out_trunk / 1e9:9.2f}G  "
          f"{src_trunk / max(out_trunk, 1):5.2f}x")
    if geo.mlp_inter:
        for frac in (1.0, 0.5, 0.25):
            f16 = geo.mlp_inter * geo.hidden * 3 * 2 * frac
            i4 = f16 / 4
            print(f"  MLP prefix at frac {frac:<4}: {f16 / 1e9:5.2f} GB now -> "
                  f"{i4 / 1e9:5.2f} GB packed")
    if skipped:
        print(f"\nskipped {len(skipped)} dense tensor(s) by --components:")
        for name, why in skipped[:8]:
            print(f"  {name}: {why}")
        if len(skipped) > 8:
            print(f"  ... and {len(skipped) - 8} more")
    print("\nnote: HOT_FRAC narrows to the FIRST Ik rows/cols, so the hot prefix is the")
    print("      file order — run with --profile hot_profile.json to write the sorted")
    print("      prefix in the same pass (this is what makes the dial's contract hold,")
    print("      and it replaces the separate 50 GB hot_sort output).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--components", default="lmhead,shexp,attn,dnproj,dnout,router")
    ap.add_argument("--profile")
    ap.add_argument("--block-rows", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--emit-dequant")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    comps = {c.strip() for c in a.components.split(",") if c.strip()}
    bad = comps - {"lmhead", "shexp", "attn", "dnproj", "dnout", "router"}
    if bad:
        die(f"unknown --components entries {sorted(bad)}")
    geo = Geo(a.src)
    files, headers, owner = build_index(a.src)
    dstart_of = {f: headers[f][1] for f in files}
    targets, skipped = plan_targets(geo, comps, files, headers, owner, dstart_of)
    if not targets:
        die(f"no packable dense tensors in {a.src} (components={sorted(comps)}; "
            f"{len(skipped)} skipped) — run: python3 tools/st_dump.py {a.src} dense")
    orders = load_profile(a.profile, geo) if a.profile else {}
    if orders:
        have = {t["layer"] for t in targets.values() if t["cls"] == "shexp"}
        miss = sorted(set(orders) - have)
        if miss:
            note(f"profile covers layers with no packable MLP tensor: {miss[:6]}"
                 f"{'...' if len(miss) > 6 else ''} — those layers would serve UNSORTED")
        unsorted_layers = sorted(have - set(orders))
        if unsorted_layers:
            note(f"WARNING: {len(unsorted_layers)} MLP layer(s) are NOT in the profile "
                 f"({unsorted_layers[:6]}{'...' if len(unsorted_layers) > 6 else ''}) — "
                 f"those layers are written UNSORTED, so HOT_FRAC<1.0 on them is random "
                 f"neuron dropout (the documented arm-E trap). Recalibrate to cover every "
                 f"layer, or pack without --profile and serve with HOT_FRAC=1.0.")
    if a.dry_run:
        dry_run(geo, targets, skipped, files, headers)
        return
    if not a.out:
        die("an output directory is required unless --dry-run")
    if os.path.isdir(a.out) and os.listdir(a.out) and not a.force:
        die(f"{a.out} is not empty — refusing to overwrite (use --force, or pick a fresh "
            f"dir; the packer never edits a container in place)")
    os.makedirs(a.out, exist_ok=True)
    if a.emit_dequant:
        os.makedirs(a.emit_dequant, exist_ok=True)
    for f in os.listdir(a.src):
        if not f.endswith(".safetensors"):
            shutil.copyfile(os.path.join(a.src, f), os.path.join(a.out, f))
            if a.emit_dequant:
                shutil.copyfile(os.path.join(a.src, f), os.path.join(a.emit_dequant, f))
    note(f"{len(targets)} dense tensor(s) -> int4 across {len(files)} shard(s); "
         f"block {a.block_rows} rows; components {sorted(comps)}"
         f"{'; profile ' + a.profile if orders else ''}")
    allstats = []
    out_bytes = 0
    for f in files:
        ts = {n: t for n, t in targets.items() if t["shard"] == f}
        hdr = headers[f][0]
        print(f"shard {f}: {len(ts)} target(s)", flush=True)
        st, nbytes = pack_shard(a.src, f, hdr, ts, geo, a, orders, a.emit_dequant)
        allstats += st
        out_bytes += nbytes
    src_trunk = sum(t["nbytes"] for t in targets.values())
    out_trunk = sum(t["I"] * t["O"] // 2 + t["O"] * 4 for t in targets.values())
    saved = src_trunk - out_trunk
    manifest = dict(
        tool="pack_trunk_i4.py", src=os.path.abspath(a.src), out=os.path.abspath(a.out),
        components=sorted(comps), block_rows=a.block_rows,
        profile=a.profile if orders else None,
        sorted_layers=sorted(orders) if orders else [],
        geometry=dict(hidden=geo.hidden, mlp_inter=geo.mlp_inter, vocab=geo.vocab,
                      n_experts=geo.n_experts, q_heads=geo.q_heads, kv_heads=geo.kv_heads,
                      q_head_dim=geo.q_head_dim, k_head_dim=geo.k_head_dim, o_in=geo.o_in,
                      dn_conv_dim=geo.dn_conv_dim, dn_vdim_tot=geo.dn_vdim_tot),
        totals=dict(tensors=len(allstats), src_bytes=src_trunk, out_bytes=out_trunk,
                    saved_bytes=saved,
                    src_gb=round(src_trunk / 1e9, 3), out_gb=round(out_trunk / 1e9, 3)),
        tensors=allstats,
        serving_note=("serve with TRUNK_MAP=1 (the mapped int4 branch dispatches on "
                      "dtype U8 + nbytes == I*O/2); HOT_FRAC narrows to the first Ik "
                      "rows/cols, so a sorted prefix needs --profile at pack time"),
    )
    if a.emit_dequant:
        manifest["emit_dequant"] = os.path.abspath(a.emit_dequant)
    if a.hash:
        manifest["sha256"] = {}
        for f in files:
            manifest["sha256"][f] = dict(
                src=_sha256(os.path.join(a.src, f)), out=_sha256(os.path.join(a.out, f)),
                dequant=(_sha256(os.path.join(a.emit_dequant, f)) if a.emit_dequant else None))
    print(f"\npacked {len(allstats)} tensor(s): {src_trunk / 1e9:.2f} GB -> "
          f"{out_trunk / 1e9:.2f} GB (saved {saved / 1e9:.2f} GB, "
          f"{src_trunk / max(out_trunk, 1):.2f}x)")
    if a.verify:
        checks, same, fails = verify_container(a.out, a.src, targets, geo)
        manifest["verify"] = dict(checks=checks, passthrough_compared=same, fails=fails)
        print(f"verify: {checks} check(s); passthrough byte-compared {same}; "
              f"{len(fails)} failure(s)")
        for x in fails:
            print(f"  FAIL {x}")
        if fails:
            json.dump(manifest, open(os.path.join(a.out, "trunk_i4_manifest.json"), "w"), indent=1)
            die("verification failed — the output container is NOT servable as int4")
        print("verify OK: every target is U8 with nbytes == I*O/2 and a F32 .qs of numel O")
    json.dump(manifest, open(os.path.join(a.out, "trunk_i4_manifest.json"), "w"), indent=1)
    print(f"manifest: {os.path.join(a.out, 'trunk_i4_manifest.json')}")


def _sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    main()
