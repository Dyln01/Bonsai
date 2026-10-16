#!/usr/bin/env python3
"""hot_sort.py — permute a container's MLP neurons by a measured hotness profile.
v2: derives matrix dims from qwen36_meta.json (hidden/shared_inter); packed tensors carry
1-D shapes in the safetensors header, so shape cannot be trusted for them."""
import json, os, struct, sys, shutil
import numpy as np

def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        tensors = {}
        for name, e in hdr.items():
            off, ln = e["data_offsets"]
            f.seek(8 + n + off)
            raw = f.read(ln)
            dt = e["dtype"]
            if dt == "F32":  arr = np.frombuffer(raw, dtype=np.float32).copy()
            elif dt == "F16": arr = np.frombuffer(raw, dtype=np.float16).copy()
            elif dt in ("U8", "I8"): arr = np.frombuffer(raw, dtype=np.uint8).copy()
            else: arr = raw
            tensors[name] = (arr, dt)
        return tensors

def write_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt, "shape": [arr.size], "data_offsets": [off, off+len(b)]}
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

def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    src, prof_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    prof = json.load(open(prof_path))
    meta = json.load(open(os.path.join(src, "qwen36_meta.json")))
    H = meta["hidden"]; I = meta["shared_inter"] or meta.get("moe_inter", 0)
    if not I:
        print("container has no MLP width (shared_inter/moe_inter) — nothing to sort"); sys.exit(1)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".json"):
            shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
    n_perm = 0
    for f in sorted(os.listdir(src)):
        if not f.endswith(".safetensors"): continue
        tensors = read_st(os.path.join(src, f))
        layers = sorted({int(n.split(".")[2]) for n in tensors if n.startswith("model.layers.")})
        for li in layers:
            key = f"L{li}"
            if key not in prof: continue
            v = np.array(prof[key]["energy"], dtype=np.float64)
            if len(v) != I:
                continue   # width mismatch: profile from a different geometry
            order = np.argsort(-v, kind="stable")
            p = f"model.layers.{li}."
            for nm in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                k = p + nm
                if k not in tensors: continue
                arr, dt = tensors[k]
                if dt in ("U8", "I8"):
                    e = unpack_i4(arr).reshape(I, H)
                    e = e[order]
                    tensors[k] = (pack_i4(e.reshape(-1)), dt)
                elif dt == "F16":
                    tensors[k] = (arr.reshape(I, H)[order].reshape(-1), dt)
            k = p + "mlp.down_proj.weight"
            if k in tensors:
                arr, dt = tensors[k]
                if dt in ("U8", "I8"):
                    e = unpack_i4(arr).reshape(H, I)
                    e = e[:, order]
                    tensors[k] = (pack_i4(e.reshape(-1)), dt)
                elif dt == "F16":
                    tensors[k] = (arr.reshape(H, I)[:, order].reshape(-1), dt)
            # scales: gate/up qs are per-ROW of [I,H] -> permute rows; down qs are per-row of [H,I] -> unchanged
            for nm in ("mlp.gate_proj.weight.qs", "mlp.up_proj.weight.qs"):
                k = p + nm
                if k in tensors:
                    arr, dt = tensors[k]
                    if arr.size == I:
                        tensors[k] = (arr[order], dt)
            n_perm += 1
        write_st(os.path.join(out, f), tensors)
        print(f"{f}: {len(layers)} layer(s)", flush=True)
    print(f"done: {n_perm} layers permuted -> {out}")

if __name__ == "__main__":
    main()
