import os
#!/usr/bin/env python3
"""Fix the F32 norm tensors in synth_qwen27: the generator's save_safetensors
wrote interleaved dicts where later .qs entries shifted offsets — the norm
tensors ended up with 8192-byte spans for 1024-element F32 arrays. This is a
generator-side bug: `data_offsets` must be contiguous and match numel*esz
exactly (st.h enforces it for float dtypes). Rebuild the file with correct
offsets from the raw blobs."""
import json, struct, collections
_Q36_HOME = os.environ.get("Q36_HOME", "/home/dylan")  # portability: override per host

P = f"{_Q36_HOME}/synth_models/synth_qwen27/model-00000.safetensors"
f = open(P, "rb")
n = struct.unpack("<Q", f.read(8))[0]
hdr_raw = json.loads(f.read(n), object_pairs_hook=collections.OrderedDict)

blobs = {}
for k, v in hdr_raw.items():
    if k == "__metadata__":
        continue
    o0, o1 = v["data_offsets"]
    f.seek(8 + n + o0)
    blobs[k] = f.read(o1 - o0)
f.close()

# rewrite with correct contiguous offsets
out = collections.OrderedDict()
off = 0
payload = []
for k, b in blobs.items():
    v = hdr_raw[k]
    numel = 1
    for d in v["shape"]:
        numel *= d
    esz = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1}[v["dtype"]]
    correct_len = numel * esz
    if len(b) != correct_len:
        print(f"fixing {k}: span {len(b)} -> {correct_len}")
        b = b[:correct_len] if len(b) > correct_len else b + b"\x00" * (correct_len - len(b))
    out[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + len(b)]}
    payload.append(b)
    off += len(b)

hj = json.dumps(out, separators=(",", ":")).encode()
hj += b" " * ((8 - len(hj) % 8) % 8)
with open(P, "wb") as g:
    g.write(struct.pack("<Q", len(hj)))
    g.write(hj)
    for b in payload:
        g.write(b)
print("rewritten OK")
