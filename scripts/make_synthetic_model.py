#!/usr/bin/env python3
"""make_synthetic_model.py — synthetic colibri-container writer (RECONSTRUCTED).

The original module lived outside this repo (`~/colibri-research/synthgen/`) and was
never committed — ROADMAP item E2. Reconstructed 2026-10-19 from its exact contract
of use in `scripts/gen_qwen27_synth.sh`:

    from make_synthetic_model import save_safetensors, quant_int4_grouped, quant_int8_row

and from the safetensors conventions proven by the in-repo generators
(`save_st` in gen_student_i4.py / gen_coder30_synth.py).

Fixed vs the original: the legacy writer emitted interleaved `.qs` entries whose
`data_offsets` could go non-contiguous (the bug documented in
`scripts/fix_qwen27_container.py`). Here offsets are ALWAYS contiguous and match
numel*itemsize exactly, so `fix_qwen27_container.py` is a no-op on fresh output.

Container formats (colibri `st.h`): fmt=4 int4 group-scaled (U8 packed, low nibble
even) with `<name>.qs` F32 scales, fmt=1 int8 per-row (I8 + `<name>.qs` F32),
F16/F32 passthrough. Group size default 128 (the REPORT.md fmt=4 int4-gs128
container class); pass gs=64 for the coder30-style expert shards.
"""
import json
import struct
import numpy as np


def save_safetensors(path, tensors, metadata=None):
    """tensors: dict name -> (ndarray, dtype_str in {"F32","F16","I8","U8"}).
    Offsets are contiguous by construction; header is 8-byte length prefixed."""
    header, blobs, off = {}, [], 0
    if metadata:
        header["__metadata__"] = metadata
    for name, (arr, dt) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def quant_int4_grouped(w, gs=128):
    """w: [O, I] f32/f16 -> (packed U8 [O, I//2], scales F32 [O, ceil(I/gs)]).

    Symmetric int4 (amax/7 per group, clip ±7, RNE), nibbles packed low-first
    within each row; I must be even; the last group of a row may be partial
    (scale still one per ceil(I/gs) slot, engine reads it the same)."""
    w = np.asarray(w, dtype=np.float32)
    O, I = w.shape
    assert I % 2 == 0, "odd inner dim not packable"
    ng = (I + gs - 1) // gs
    pad = ng * gs - I
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = wp.reshape(O, ng, gs)
    amax = np.abs(g).max(axis=2, keepdims=True)
    s = np.maximum(amax / 7.0, 1e-8).astype(np.float32).reshape(O, ng)
    q = np.rint(np.clip(g / np.maximum(amax, 1e-8) * 7.0, -7, 7)).astype(np.int8)
    q = q.reshape(O, ng * gs)[:, :I].reshape(-1)
    lo = (q[0::2] & 0xF).astype(np.uint8)
    hi = ((q[1::2] & 0xF) << 4).astype(np.uint8)
    return (lo | hi), s


def quant_int8_row(w):
    """w: [O, I] -> (q I8 [O, I], scales F32 [O]); per-row amax/127, RNE."""
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / 127.0, 1e-8).astype(np.float32).reshape(-1)
    q = np.rint(np.clip(w / np.maximum(amax, 1e-8) * 127.0, -127, 127)).astype(np.int8)
    return q, s


if __name__ == "__main__":  # tiny self-check (CI runs this)
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((64, 256)) * 0.05).astype(np.float32)
    q4, s4 = quant_int4_grouped(w, gs=128)
    assert q4.shape == (64 * 256 // 2,) and s4.shape == (64, 2)  # flat packed = engine convention
    # dequant: byte i carries elements 2i (low nibble) and 2i+1 (high nibble)
    lo = (q4 & 0xF).astype(np.int8)
    hi = ((q4 >> 4) & 0xF).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo)
    hi = np.where(hi > 7, hi - 16, hi)
    up = np.empty((64, 256), np.int8)
    up.reshape(-1)[0::2] = lo
    up.reshape(-1)[1::2] = hi
    deq = up.astype(np.float32) * np.repeat(s4, 128, axis=1)
    rel = np.linalg.norm(deq - w) / np.linalg.norm(w)
    # plain int4-gs128 on Gaussian weights runs ~0.10-0.12 rel L2 (cf. rotated-int4
    # 0.096 in docs/KVTQ_QWEN36.md); a wrong nibble/scale layout lands near 1.0+
    assert rel < 0.15, rel
    q8, s8 = quant_int8_row(w)
    assert q8.shape == w.shape and s8.shape == (64,)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.safetensors")
        save_safetensors(p, {"a": (q4, "U8"), "a.qs": (s4, "F32"),
                             "b": (q8, "I8"), "b.qs": (s8, "F32"),
                             "n": (np.ones(8, np.float16), "F16")})
        n = struct.unpack("<Q", open(p, "rb").read(8))[0]
        hdr = json.loads(open(p, "rb").read()[8:8 + n])
        end = 0
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            assert v["data_offsets"][0] == end, k   # contiguous
            end = v["data_offsets"][1]
        assert os.path.getsize(p) == 8 + n + end
    print("make_synthetic_model self-check OK (int4-gs128 rel err %.4f)" % rel)
