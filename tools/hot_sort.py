#!/usr/bin/env python3
"""hot_sort.py — permute a container's MLP neurons by a measured hotness profile.
v2: derives matrix dims from qwen36_meta.json (hidden/shared_inter); packed tensors carry
1-D shapes in the safetensors header, so shape cannot be trusted for them.
v2.1: FIXES the v2 reader bug behind the q27 mini-PC crash — safetensors
data_offsets are [begin, END), not [offset, length]; v2 read `end` BYTES from
`begin`, inflating every tensor that is not first in its shard by `begin` bytes
(the real container's F16 gate_proj at begin=178,268,160 came back as
(2*17408+1)*5120 elements — the exact reshape ValueError). Reads are now checked
the way c/st.h checks them (float dtypes: span == numel*esz; short reads and
out-of-file spans refused; unknown dtypes refused at the entry, known opaque
dtypes passed through verbatim), cross-shard duplicate tensor names follow
st_init's rule (model.safetensors.index.json resolves, anything else is a
refusal), every permute target is size-checked against the meta geometry BEFORE
reshaping, and a file whose profiled layers yield zero permuted tensors warns
loudly — a silent no-op sort would invalidate the arm-E measurements.
(scripts/test_hot_sort.py)."""
import json, os, struct, sys, shutil
import numpy as np

# st.h st_dtype_code/st_dtype_esz, mirrored (tools/st_dump.py keeps the same table).
ESZ = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1,
       "F8_E4M3": 1, "F8_E4M3FN": 1, "float8_e4m3fn": 1,
       "F8_E8M0": 1, "F8_E8M0FNU": 1, "I64": 8, "U64": 8, "I32": 4, "U32": 4}
NUMPY_DT = {"F32": np.float32, "F16": np.float16, "U8": np.uint8, "I8": np.uint8}

_warned = set()

def die(msg):
    print(f"hot_sort: error: {msg}", file=sys.stderr)
    sys.exit(2)

def warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(f"hot_sort: WARNING: {msg}", file=sys.stderr)

def header_names(path):
    """(ordered unique tensor names, names duplicated WITHIN this one header)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    keys = [k for k, _ in json.loads(raw, object_pairs_hook=lambda p: p)
            if k != "__metadata__"]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    return list(dict.fromkeys(keys)), dups

def read_st(path):
    """-> (tensors {name: (arr, dt, shape)}, metadata). arr is numpy for
    F32/F16/U8/I8 (the dtypes hot_sort permutes) and raw bytes for the other
    known dtypes (BF16/U32/I64/F8: copied through untouched)."""
    fsz = os.path.getsize(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        metadata = hdr.get("__metadata__")
        tensors = {}
        for name, e in hdr.items():
            if name == "__metadata__":
                continue
            dt, shape, offs = e.get("dtype"), e.get("shape"), e.get("data_offsets")
            if not isinstance(dt, str) or not isinstance(shape, list) or \
                    not (isinstance(offs, list) and len(offs) == 2):
                die(f"{path}: tensor '{name}': malformed header entry {e!r}")
            if dt not in ESZ:
                die(f"{path}: tensor '{name}': unsupported dtype '{dt}' — the engine's "
                    f"st_dtype_code refuses this container too")
            try:
                begin, end = int(offs[0]), int(offs[1])
            except (TypeError, ValueError):
                die(f"{path}: tensor '{name}': non-integer data_offsets {offs!r}")
            ln = end - begin     # data_offsets are [begin, END) — v2 read `end` as a LENGTH
            numel = 1
            for d in shape:
                numel *= int(d)
            esz = ESZ[dt]
            # st_init's cross-check: every dtype except raw U8/I8 quant bytes must
            # have span == numel*esz. Refusing here keeps hot_sort's view of the
            # container identical to the engine's (load_t_n checks the shape-derived
            # numel; this tool reads the byte span — on a sane file they agree).
            if dt not in ("U8", "I8") and ln != numel * esz:
                die(f"{path}: tensor '{name}': dtype {dt} shape {shape} = {numel} "
                    f"elements = {numel * esz:,} bytes, but data_offsets span {ln:,} "
                    f"bytes. st_init refuses float tensors like this ('numel disagrees "
                    f"with byte span'), so this shard was not written by the pinned "
                    f"convert_qwen36.py. Run: python3 tools/st_dump.py <container_dir> "
                    f"and share the output.")
            if begin < 0 or ln < 0 or 8 + n + end > fsz:
                die(f"{path}: tensor '{name}': data_offsets [{begin},{end}) outside the "
                    f"data region of a {fsz:,}-byte file — truncated or corrupt shard")
            f.seek(8 + n + begin)
            raw = f.read(ln)
            if len(raw) != ln:
                die(f"{path}: tensor '{name}': short read ({len(raw):,} of {ln:,} bytes) "
                    f"— truncated or corrupt shard")
            if dt in NUMPY_DT:
                arr = np.frombuffer(raw, dtype=NUMPY_DT[dt]).copy()
            else:
                arr = raw                      # opaque passthrough
            tensors[name] = (arr, dt, list(shape))
        return tensors, metadata

def write_st(path, tensors, metadata=None):
    header, blobs, off = {}, [], 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (arr, dt, shape0) in tensors.items():
        if isinstance(arr, (bytes, bytearray)):
            b = bytes(arr); shape = list(shape0)      # passthrough: keep source shape
        else:
            b = arr.tobytes(); shape = [arr.size]     # v2 contract: flat 1-D shapes
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off+len(b)]}
        blobs.append(b); off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for b in blobs: f.write(b)

def unpack_i4(flat):
    # byte k: LOW nibble = element 2k, HIGH nibble = element 2k+1 (converter convention)
    lo = (flat & 0xF).astype(np.int16)
    hi = (flat >> 4).astype(np.int16)
    n = flat.size * 2
    e = np.empty(n, dtype=np.int16)
    e[0::2] = np.where(lo & 8, lo - 16, lo)
    e[1::2] = np.where(hi & 8, hi - 16, hi)
    return e.astype(np.int8)

def pack_i4(vals):
    q = np.clip(vals, -8, 7).astype(np.int16)
    lo = (q[0::2] & 0xF).astype(np.uint8)
    hi = ((q[1::2] & 0xF) << 4).astype(np.uint8)
    return (lo | hi)

def geometry_die(f, k, dt, size, want, cols, extra=""):
    rows_got = size // cols if cols and size % cols == 0 else None
    detail = f" = {rows_got} rows of {cols}" if rows_got else ""
    die(f"{f}: {k}: {dt} tensor has {size:,} elements/bytes{detail}; the meta geometry "
        f"(shared_inter x hidden) implies {want:,}.{extra} The engine's load_t_n refuses "
        f"this tensor too (numel != shared_inter*hidden), so this shard was not written "
        f"by the pinned convert_qwen36.py dense path. Run: python3 tools/st_dump.py "
        f"<container_dir> mlp and share the output.")

def build_owner_map(src, files, weight_map):
    """name -> authoritative shard, mirroring st_init's duplicate rule: a name in
    two shards is read from the copy model.safetensors.index.json names (the other
    is ignored); no index, or an index that doesn't resolve, is a refusal."""
    owner = {}
    for f in files:
        names, intra_dups = header_names(os.path.join(src, f))
        for k in intra_dups:
            if weight_map.get(k) != f:
                die(f"{f}: tensor '{k}' appears twice inside the same header JSON and "
                    f"model.safetensors.index.json does not resolve it to this file — "
                    f"st_init refuses this (last-copy-wins applies only when the index "
                    f"names this shard)")
        for name in names:
            prev = owner.get(name)
            if prev is None or prev == f:
                owner[name] = f
                continue
            auth = weight_map.get(name)
            if auth == f:
                owner[name] = f          # overlay shard wins (st_init replace_idx)
            elif auth == prev:
                pass                     # index keeps the first copy; this one ignored
            else:
                why = (f"the index maps it to '{auth}', a third file" if auth else
                       ("the index does not list it" if weight_map else
                        "there is no model.safetensors.index.json"))
                die(f"tensor '{name}' is defined in both '{prev}' and '{f}' and {why} "
                    f"— st_init refuses to load a directory like this; remove the stray "
                    f"copy (run: python3 tools/st_dump.py <container_dir>)")
    return owner

def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    src, prof_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    prof = json.load(open(prof_path))
    meta = json.load(open(os.path.join(src, "qwen36_meta.json")))
    H = meta["hidden"]; I = meta["shared_inter"] or meta.get("moe_inter", 0)
    if not I:
        print("container has no MLP width (shared_inter/moe_inter) — nothing to sort"); sys.exit(1)
    files = [f for f in sorted(os.listdir(src)) if f.endswith(".safetensors")]
    idx_path = os.path.join(src, "model.safetensors.index.json")
    weight_map = {}
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path)).get("weight_map") or {}
    owner = build_owner_map(src, files, weight_map)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".json"):
            shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
    n_layers = n_tensors = 0
    for f in files:
        tensors, metadata = read_st(os.path.join(src, f))
        for name in [k for k in tensors if owner.get(k, f) != f]:
            del tensors[name]     # ignored duplicate copy — the engine reads the other shard
        layers = sorted({int(n.split(".")[2]) for n in tensors if n.startswith("model.layers.")})
        profiled = permuted = 0
        for li in layers:
            key = f"L{li}"
            if key not in prof: continue
            v = np.array(prof[key]["energy"], dtype=np.float64)
            if len(v) != I:
                warn_once("width", f"profile width {len(v)} != shared_inter {I} for {key} "
                                   f"— profile from a different geometry; layer skipped")
                continue   # width mismatch: profile from a different geometry
            profiled += 1
            order = np.argsort(-v, kind="stable")
            p = f"model.layers.{li}."
            for nm in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                k = p + nm
                if k not in tensors: continue
                arr, dt, shp = tensors[k]
                if dt in ("U8", "I8"):
                    if arr.size != I * H // 2:
                        geometry_die(f, k, f"packed {dt}", arr.size, I * H // 2, H // 2,
                                     " Int4-packed bytes = shared_inter*hidden/2.")
                    e = unpack_i4(arr).reshape(I, H)
                    e = e[order]
                    tensors[k] = (pack_i4(e.reshape(-1)), dt, shp); permuted += 1
                elif dt == "F16":
                    hint = ""
                    if arr.size == (2 * I + 1) * H:
                        hint = (" Size = (2*shared_inter+1)*hidden — a fused"
                                " [gate;up;+1 row] layout nothing in the pinned"
                                " toolchain writes.")
                    elif arr.size == 2 * I * H:
                        hint = " Size = 2*shared_inter*hidden — a fused [gate;up] layout."
                    if arr.size != I * H:
                        geometry_die(f, k, dt, arr.size, I * H, H, hint)
                    tensors[k] = (arr.reshape(I, H)[order].reshape(-1), dt, shp); permuted += 1
                else:
                    die(f"{f}: {k}: dtype {dt} — hot_sort permutes F16 and int4-packed "
                        f"U8/I8 MLP tensors only (the pinned converter's dense formats); "
                        f"passing anything else through unsorted would silently "
                        f"invalidate the sorted container")
            k = p + "mlp.down_proj.weight"
            if k in tensors:
                arr, dt, shp = tensors[k]
                if dt in ("U8", "I8"):
                    if arr.size != H * I // 2:
                        geometry_die(f, k, f"packed {dt}", arr.size, H * I // 2, I // 2,
                                     " Int4-packed bytes = hidden*shared_inter/2.")
                    e = unpack_i4(arr).reshape(H, I)
                    e = e[:, order]
                    tensors[k] = (pack_i4(e.reshape(-1)), dt, shp); permuted += 1
                elif dt == "F16":
                    if arr.size != H * I:
                        geometry_die(f, k, dt, arr.size, H * I, I)
                    tensors[k] = (arr.reshape(H, I)[:, order].reshape(-1), dt, shp); permuted += 1
                else:
                    die(f"{f}: {k}: dtype {dt} — see the gate/up dtype rule above")
            # scales: gate/up qs are per-ROW of [I,H] -> permute rows; down qs are per-row of [H,I] -> unchanged
            for nm in ("mlp.gate_proj.weight.qs", "mlp.up_proj.weight.qs"):
                k = p + nm
                if k in tensors:
                    arr, dt, shp = tensors[k]
                    if not isinstance(arr, (bytes, bytearray)) and arr.size == I:
                        tensors[k] = (arr[order], dt, shp)
                    else:
                        nsc = arr.size if hasattr(arr, "size") else len(arr)
                        warn_once("qs", f"{k}: {nsc} scale elements != shared_inter {I} — "
                                        f"group-wise scales cannot be row-permuted (and the "
                                        f"engine's int4 map dialect reads per-row qs only); "
                                        f"left UNSORTED")
        n_layers += profiled
        n_tensors += permuted
        if profiled and not permuted:
            warn_once("noop", f"{f}: {profiled} profiled layer(s) but NO "
                              f"mlp.{{gate,up,down}}_proj tensors matched — nothing was "
                              f"permuted in this shard (fused or shared_expert naming? "
                              f"run: python3 tools/st_dump.py <container_dir> mlp). "
                              f"Serving this output as 'sorted' would silently measure "
                              f"an UNSORTED container (arm E).")
        write_st(os.path.join(out, f), tensors, metadata)
        print(f"{f}: {len(layers)} layer(s), {permuted} tensor(s) permuted", flush=True)
    print(f"done: {n_tensors} tensor(s) permuted across {n_layers} profiled layer(s) -> {out}")

if __name__ == "__main__":
    main()
