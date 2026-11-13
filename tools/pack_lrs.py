#!/usr/bin/env python3
"""pack_lrs.py — emit a real LRS container (U, V, S_idx, S_val), or refuse to.

Step 2 of the LRS plan. The requirements are the caller's, and each one is enforced here
rather than documented:

  1. it actually READS the source tensors        (reuses pack_trunk_i4's readers)
  2. it emits FOUR real tensors per target       (<name>.lrs_U, .lrs_V, .lrs_S_idx, .lrs_S_val)
  3. meta["lrs"] lists ONLY tensors whose files exist on disk
  4. it REFUSES to produce a container that advertises LRS without the files
     (--verify re-reads what it wrote and fails otherwise; --verify-only audits any container)
  5. safetensors, the dialect the engine already reads (no raw .tofile sidecars)
  6. no engine work: the emitted container is not loadable by stock, and says so

Plus the gate that makes Step 2 safe to have built before Step 1's answer: **a tensor is only
packed when its LRS beats int4 at EQUAL BYTES on that tensor.** The byte model is the one
analyze_lrs.py uses — int4 `m*n/2 + m*4`, LRS `(m+n)*r*2 + nnz*6` with the sparse budget
being the bytes LEFT of int4's count — so the packer cannot ship something that would have
lost the analysis. `--allow-losing` overrides it for experiments and says what it overrode.

Usage:
  pack_lrs.py --model SRC --out DST --rank 64 [--classes deltanet,lm_head,attn] [--verify]
  pack_lrs.py --verify-only DST
  pack_lrs.py --self-test
"""
import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pack_trunk_i4 as pk
import analyze_lrs as al

META = "qwen36_meta.json"
COPY_ALWAYS = [META, "config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "model.safetensors.index.json"]
SUF = {"U": ".lrs_U", "V": ".lrs_V", "idx": ".lrs_S_idx", "val": ".lrs_S_val"}


def plan_tensor(W, rank, residual_cap_bytes):
    """-> (U, V, S_idx, S_val, info) or (None, ..., info) when it does not qualify."""
    m, n = W.shape
    int4_bytes = m * n // 2 + m * 4
    low_bytes = (m + n) * rank * 2
    nnz = int((int4_bytes - low_bytes) // 6)
    capped = False
    if nnz > 0 and nnz * 6 > residual_cap_bytes:
        nnz = int(residual_cap_bytes // 6)          # stay inside --residual-cap-mb
        capped = True
    U, S, Vt = al.rand_svd(W, rank)
    Ar = (U * S) @ Vt
    R = W - Ar
    U16 = U.astype(np.float16)
    V16 = Vt.T.astype(np.float16)
    if nnz > 0:
        k = int(min(nnz, R.size))
        flat = np.abs(R).ravel()
        idx = (np.argpartition(flat, -k)[-k:] if k < flat.size else np.arange(flat.size))
        S_idx = idx.astype(np.uint32)
        S_val = R.ravel()[idx].astype(np.float16)
        Wh = Ar + (np.zeros(R.size, np.float32).reshape(-1) * 0 +
                   (lambda z: (z.__setitem__(idx, R.ravel()[idx]), z)[1])(np.zeros(R.size, np.float32))
                   ).reshape(R.shape)
    else:
        S_idx = np.zeros(0, np.uint32)
        S_val = np.zeros(0, np.float16)
        Wh = Ar
    err = al.rel_l2(W, Wh)
    err_i4 = al.rel_l2(W, al.int4_roundtrip(W)[0])
    bytes_used = low_bytes + S_idx.size * 6
    info = dict(shape=[m, n], rank=int(rank), nnz=int(S_idx.size), int4_bytes=int(int4_bytes),
                lrs_bytes=int(bytes_used), err=err, int4_err=err_i4,
                ratio=err / max(err_i4, 1e-30), capped=capped,
                saved_bytes=int(int4_bytes - bytes_used))
    if info["ratio"] >= 1.0:
        return None, None, None, None, info
    return U16, V16, S_idx, S_val, info


def write_container(model, out, targets, rank, residual_cap_bytes, allow_losing):
    files, headers, owner = pk.build_index(model)
    os.makedirs(out, exist_ok=True)
    for name in COPY_ALWAYS:
        src = os.path.join(model, name)
        if os.path.exists(src) and os.path.dirname(src) != os.path.abspath(out):
            shutil.copy2(src, os.path.join(out, name))

    lrs_meta, decisions, saved_total = {}, [], 0
    for name in sorted(owner):
        cl = pk.classify(name)
        if cl is None or cl[1] not in targets:
            continue
        shard = owner[name]
        hdr, dstart = headers[shard]
        ent = hdr[name]
        dtype, shape = ent["dtype"], ent["shape"]
        if dtype in ("U8", "I8") or len(shape) != 2:
            decisions.append((name, f"SKIP: {dtype} shape {shape} (already packed / not 2-D)"))
            continue
        m, n = int(shape[0]), int(shape[1])
        with open(os.path.join(model, shard), "rb") as f:
            W = pk.read_rows(f, dstart + ent["data_offsets"][0], m, n, pk.ESZ[dtype])
        U, V, S_idx, S_val, info = plan_tensor(W, rank, residual_cap_bytes)
        if U is None and not allow_losing:
            decisions.append((name, f"REFUSED: LRS {info['err']:.4f} vs int4 {info['int4_err']:.4f} "
                                    f"at equal bytes (x{info['ratio']:.2f}) — the analysis said no"))
            continue
        lrs_meta[name] = dict(rank=info["rank"], nnz=info["nnz"], shape=[m, n],
                              lrs_bytes=info["lrs_bytes"], int4_bytes=info["int4_bytes"],
                              err=info["err"], int4_err=info["int4_err"], ratio=info["ratio"],
                              saved_bytes=info["saved_bytes"],
                              tensors={k: name + v for k, v in SUF.items()})
        saved_total += max(info["saved_bytes"], 0)
        decisions.append((name, f"PACKED rank {rank} nnz {info['nnz']} "
                                f"x{info['ratio']:.2f} int4's error, saves "
                                f"{info['saved_bytes'] / 1e6:.2f} MB"
                                + ("  (residual capped)" if info["capped"] else "")))
        # keep the buffers for the write pass below
        lrs_meta[name]["_buf"] = (U, V, S_idx, S_val)
        lrs_meta[name]["_shard"] = shard

    for fname in files:
        hdr, dstart = headers[fname]
        entries, payload = [], []
        for name, ent in hdr.items():
            if name == "__metadata__":
                continue
            rep = lrs_meta.get(name)
            if rep is None:
                off = ent["data_offsets"][0]
                nb = ent["data_offsets"][1] - off
                entries.append((name, (ent["dtype"], nb, ent["shape"])))
                payload.append(("copy", dstart + off, nb))
                continue
            U, V, S_idx, S_val = rep["_buf"]   # four arrays (see the store above)
            for key, arr, dt in (("U", U, "F16"), ("V", V, "F16"),
                                 ("idx", S_idx, "U32"), ("val", S_val, "F16")):
                b = arr.tobytes()
                entries.append((name + SUF[key], (dt, len(b), list(arr.shape))))
                payload.append(("bytes", b, None))
        with open(os.path.join(out, fname), "wb") as fo, \
                open(os.path.join(model, fname), "rb") as fi:
            pk.write_header(fo, entries)
            for kind, a, b in payload:
                if kind == "copy":
                    pk.copy_span(fo, fi, a, b)
                else:
                    fo.write(a)

    mpath = os.path.join(out, META)
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
    if lrs_meta:
        meta["lrs"] = {"version": 1, "rank": int(rank),
                       "tensors": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                   for k, v in lrs_meta.items()}}
    elif "lrs" in meta:
        del meta["lrs"]          # never advertise LRS without a single packed tensor
    json.dump(meta, open(mpath, "w"), indent=1)

    print(f"== pack_lrs: {len(lrs_meta)} tensor(s) packed, "
          f"{saved_total / 1e6:.2f} MB/token saved vs int4 ==")
    for name, why in decisions:
        print(f"  {name}\n      {why}")
    return lrs_meta


def verify(container, rows_note=True):
    """Requirement 4: a container that advertises LRS must contain exactly those files."""
    bad = 0
    mpath = os.path.join(container, META)
    if not os.path.exists(mpath):
        print(f"FAIL  {mpath} missing")
        return 1
    meta = json.load(open(mpath))
    block = meta.get("lrs")
    if not block:
        print("note  no meta[\"lrs\"] block: nothing advertised, nothing to verify")
        return 0
    files, headers, owner = pk.build_index(container)
    present = set(owner)
    claimed = set((block.get("tensors") or {}))
    print(f"meta advertises LRS for {len(claimed)} tensor(s)")
    missing_dense, missing_files = [], []
    for name, rec in (block.get("tensors") or {}).items():
        if name in present:
            missing_dense.append(name)
        for key in ("U", "V", "idx", "val"):
            tn = rec["tensors"].get(key)
            if tn not in present:
                missing_files.append((name, key, tn))
                continue
            shard = owner[tn]
            ent = headers[shard][0][tn]
            want = {"U": (rec["shape"][0], rec["rank"]), "V": (rec["shape"][1], rec["rank"])}
            if key in want and list(ent["shape"]) != list(want[key]):
                print(f"FAIL  {tn}: shape {ent['shape']} != {want[key]}")
                bad += 1
            if key in ("idx", "val") and int(np.prod(ent["shape"])) != rec["nnz"]:
                print(f"FAIL  {tn}: {ent['shape']} != nnz {rec['nnz']}")
                bad += 1
            dt_want = "U32" if key == "idx" else "F16"
            if ent["dtype"] != dt_want:
                print(f"FAIL  {tn}: dtype {ent['dtype']} != {dt_want}")
                bad += 1
    for name in missing_dense:
        print(f"FAIL  {name} is listed in meta[\"lrs\"] but still present in the container "
              f"(its bytes were not saved)")
        bad += 1
    for name, key, tn in missing_files:
        print(f"FAIL  {name}: meta lists {key} -> {tn}, which is NOT on disk")
        bad += 1
    extra = [t for t in present if t.endswith(tuple(SUF.values())) and
             not any(t == r["tensors"][k] for r in (block.get("tensors") or {}).values()
                     for k in ("U", "V", "idx", "val"))]
    for t in extra:
        print(f"FAIL  {t} is an LRS tensor on disk that meta does not list")
        bad += 1
    if rows_note:
        print(f"note  self-consistency only: the reconstruction error in meta was measured "
              f"against the source at pack time and cannot be re-derived from this container")
    print("LRS VERIFY: " + ("PASS" if not bad else f"{bad} FAILURE(S)"))
    return 1 if bad else 0


def _fixture(tmp):
    """Two tensors: one genuinely rank-8 (must pack), one iid Gaussian (must be refused)."""
    rs = np.random.default_rng(11)
    A = rs.standard_normal((96, 8)).astype(np.float32)
    B = rs.standard_normal((8, 64)).astype(np.float32)
    low = (A @ B).astype(np.float16)
    flat = rs.standard_normal((96, 64)).astype(np.float32).astype(np.float16)
    os.makedirs(tmp, exist_ok=True)
    names = ["lm_head.weight", "model.layers.0.linear_attn.in_proj_qkv.weight"]
    arrays = [low, flat]
    entries, payload = [], []
    for nm, arr in zip(names, arrays):
        b = arr.tobytes()
        entries.append((nm, ("F16", len(b), list(arr.shape))))
        payload.append(b)
    with open(os.path.join(tmp, "model.safetensors"), "wb") as f:
        pk.write_header(f, entries)
        for b in payload:
            f.write(b)
    json.dump({"hidden": 64, "n_layers": 1}, open(os.path.join(tmp, META), "w"))
    return tmp


def self_test():
    import tempfile
    bad = 0
    def check(ok, what, detail=""):
        nonlocal bad
        print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
        if not ok:
            bad += 1
    tmp = tempfile.mkdtemp(prefix="lrs_")
    src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
    _fixture(src)
    print("== the packer must pack the low-rank tensor and REFUSE the flat one ==")
    write_container(src, out, {"lmhead", "dnproj"}, 8, 512 << 20, allow_losing=False)
    present = set(pk.build_index(out)[2])
    meta = json.load(open(os.path.join(out, META)))
    check(al.resolve_classes("lm_head,deltanet") == {"lmhead", "dnproj", "dnout"},
          "the class aliases resolve (lm_head -> lmhead, deltanet -> dnproj+dnout)",
          str(sorted(al.resolve_classes("lm_head,deltanet"))))
    check("lm_head.weight.lrs_U" in present and "lm_head.weight.lrs_V" in present
          and "lm_head.weight.lrs_S_idx" in present and "lm_head.weight.lrs_S_val" in present,
          "four LRS tensors emitted for the low-rank tensor")
    check("lm_head.weight" not in present, "the dense tensor it replaced is gone (bytes saved)")
    check(list(meta["lrs"]["tensors"]) == ["lm_head.weight"],
          "meta lists exactly the packed tensor", str(list(meta["lrs"]["tensors"])))
    check("model.layers.0.linear_attn.in_proj_qkv.weight" in present,
          "the refused tensor is left dense, untouched")
    print("== verify passes on what it wrote, and catches a lying container ==")
    check(verify(out) == 0, "verify passes on the emitted container")
    os.remove(os.path.join(out, "model.safetensors"))       # break it: advertise without files
    _fixture(out + "_json")
    shutil.copy2(os.path.join(src, "model.safetensors"), os.path.join(out, "model.safetensors"))
    check(verify(out) != 0, "verify FAILS when the advertised files are missing (req. 4)")
    meta2 = json.load(open(os.path.join(src, META)))
    meta2["lrs"] = {"version": 1, "rank": 8, "tensors": {"lm_head.weight": {
        "rank": 8, "nnz": 10, "shape": [96, 64], "tensors": {k: "x" + v for k, v in SUF.items()}}}}
    os.makedirs(out + "2", exist_ok=True)
    shutil.copy2(os.path.join(src, "model.safetensors"), os.path.join(out + "2", "model.safetensors"))
    json.dump(meta2, open(os.path.join(out + "2", META), "w"))
    check(verify(out + "2") != 0, "verify FAILS on a container that advertises LRS it lacks "
                                  "(the stub's exact defect)")
    print("\nPACK_LRS SELF-TEST: " + ("PASS" if not bad else f"{bad} FAILED"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--classes", default="deltanet,lm_head,attn")
    ap.add_argument("--residual-cap-mb", type=float, default=512.0)
    ap.add_argument("--allow-losing", action="store_true",
                    help="pack even when LRS loses to int4 at equal bytes (experiments only)")
    ap.add_argument("--verify", action="store_true", help="verify after writing")
    ap.add_argument("--verify-only")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.verify_only:
        return verify(a.verify_only)
    if not a.model or not a.out:
        ap.error("--model and --out are required (or --self-test/--verify-only)")

    # Same resolver as the analyzer: its default --classes "deltanet,lm_head,attn" used to
    # map lm_head to {"lm_head"} while the repo's class is "lmhead", so that tensor was
    # silently never packed.
    targets = al.resolve_classes(a.classes)
    if targets is None:
        targets = {"lmhead", "dnproj", "dnout", "attn", "shexp"}
    unknown = targets - {"lmhead", "dnproj", "dnout", "attn", "shexp", "router"}
    if unknown:
        raise SystemExit(f"pack_lrs: class(es) {sorted(unknown)} are not packable "
                         f"(packable: lmhead, deltanet/dnproj/dnout, attn, mlp/shexp)")
    write_container(a.model, a.out, targets, a.rank, a.residual_cap_mb * 1e6, a.allow_losing)
    if a.verify:
        return verify(a.out)
    print("note: verify it with --verify-only (always true, not just when asked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
