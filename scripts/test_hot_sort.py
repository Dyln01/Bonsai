#!/usr/bin/env python3
"""test_hot_sort.py — fixture gate for tools/hot_sort.py (v2.1) and tools/st_dump.py.

Mechanical oracles only (house rule): the permutation is checked against the
shipped argsort formula on fixture tensors with distinct rows, outputs are
re-parsed for st.h-style header self-consistency, and refusals are checked by
exit code + the actionable substring the message must carry. The fixtures
reproduce the real q27 failure classes: span/shape disagreement, a fused
(2*shared_inter+1)*hidden F16 gate/up, cross-shard duplicate names with and
without model.safetensors.index.json, opaque-dtype passthrough, and the
shared_expert-naming silent no-op.

Run: python3 scripts/test_hot_sort.py   (needs numpy; exit 0 = all pass)
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOT_SORT = os.path.join(ROOT, "tools", "hot_sort.py")
ST_DUMP = os.path.join(ROOT, "tools", "st_dump.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hot_sort  # noqa: E402  (pack_i4/unpack_i4 for the i4 fixtures)

H, I, L = 16, 8, 2          # hidden, shared_inter, layers — tiny on purpose
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


def save_st(path, entries, metadata=None):
    """entries: [(name, dtype, shape, bytes)] — an INDEPENDENT writer (does not
    reuse hot_sort.write_st, so the tests do not share a bug with the tool)."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    blobs, off = [], 0
    for name, dt, shape, b in entries:
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def read_st_raw(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for name, e in hdr.items():
            if name == "__metadata__":
                out[name] = e
                continue
            begin, end = e["data_offsets"]       # [begin, END), not [offset, length]
            f.seek(8 + n + begin)
            out[name] = (f.read(end - begin), e["dtype"], e["shape"])
    return out


def base_files(d, layers=L):
    json.dump({}, open(os.path.join(d, "config.json"), "w"))
    json.dump({"hidden": H, "shared_inter": I, "num_experts": 0},
              open(os.path.join(d, "qwen36_meta.json"), "w"))


def energies(rng):
    """Per-layer energy vectors with distinct values -> a unique argsort order."""
    return {f"L{li}": {"energy": [float(x) for x in rng.permutation(np.arange(I)) + 1]}
            for li in range(L)}


def f16_matrix(rng, rows, cols):
    return (rng.standard_normal((rows, cols)) * 0.05).astype(np.float16)


def build_f16(d, rng, extra=None):
    """Clean dense-F16 container: one shard per layer, gate/up (I,H), down (H,I)."""
    base_files(d)
    src = {}
    for li in range(L):
        p = f"model.layers.{li}."
        ent = [(p + "input_layernorm.weight", "F32", [H],
                (np.ones(H, np.float32) * (li + 1)).tobytes())]
        for nm, rows, cols in (("mlp.gate_proj.weight", I, H),
                               ("mlp.up_proj.weight", I, H),
                               ("mlp.down_proj.weight", H, I)):
            m = f16_matrix(rng, rows, cols)
            src[p + nm] = m
            ent.append((p + nm, "F16", [rows, cols], m.tobytes()))
        if extra:
            ent += extra(li)
        save_st(os.path.join(d, f"model-{li:05d}.safetensors"), ent)
    return src


def build_prof(d, prof):
    json.dump(prof, open(os.path.join(d, "prof.json"), "w"))


def run(tool, *args):
    r = subprocess.run([sys.executable, tool, *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def headers_consistent(path):
    """st_init's rule: for every non-U8/I8 entry, span == numel*esz."""
    ESZ = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "U32": 4, "I64": 8}
    for name, (b, dt, shape) in read_st_raw(path).items():
        if name == "__metadata__":
            continue
        numel = 1
        for s in shape:
            numel *= s
        if dt not in ("U8", "I8") and len(b) != numel * ESZ[dt]:
            return False, name
    return True, None


def order_of(prof, li):
    v = np.array(prof[f"L{li}"]["energy"], dtype=np.float64)
    return np.argsort(-v, kind="stable")


# ---------------------------------------------------------------- fixtures/tests
def test_f16_clean():
    print("== f16 clean container: permutation oracle, determinism, header sanity")
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        src_d, out_a, out_b = f"{td}/c", f"{td}/a", f"{td}/b"
        os.makedirs(src_d)
        src = build_f16(src_d, rng)
        prof = energies(rng)
        build_prof(src_d, prof)
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out_a)
        check("sort rc=0", rc == 0, msg[-400:])
        ok = True
        for li in range(L):
            order = order_of(prof, li)
            got = read_st_raw(f"{out_a}/model-{li:05d}.safetensors")
            p = f"model.layers.{li}."
            g = np.frombuffer(got[p + "mlp.gate_proj.weight"][0], np.float16).reshape(I, H)
            u = np.frombuffer(got[p + "mlp.up_proj.weight"][0], np.float16).reshape(I, H)
            dn = np.frombuffer(got[p + "mlp.down_proj.weight"][0], np.float16).reshape(H, I)
            ok &= np.array_equal(g, src[p + "mlp.gate_proj.weight"][order])
            ok &= np.array_equal(u, src[p + "mlp.up_proj.weight"][order])
            ok &= np.array_equal(dn, src[p + "mlp.down_proj.weight"][:, order])
            norm = np.frombuffer(got[p + "input_layernorm.weight"][0], np.float32)
            ok &= np.array_equal(norm, np.ones(H, np.float32) * (li + 1))
        check("gate/up rows + down cols permuted by profile order; norms untouched", ok)
        for li in range(L):
            good, bad = headers_consistent(f"{out_a}/model-{li:05d}.safetensors")
            check(f"output shard {li} header self-consistent", good, str(bad))
        rc2, _ = run(HOT_SORT, src_d, f"{src_d}/prof.json", out_b)
        same = rc2 == 0 and all(
            open(f"{out_a}/model-{li:05d}.safetensors", "rb").read() ==
            open(f"{out_b}/model-{li:05d}.safetensors", "rb").read() for li in range(L))
        check("deterministic: second run byte-identical", same)
        rc3, msg3 = run(ST_DUMP, src_d)
        check("st_dump clean container rc=0", rc3 == 0, msg3[-300:])


def test_i4_clean():
    print("== int4-packed container: unpack/permute/repack oracle, qs rows, qs-width warning")
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)
        base_files(src_d)
        prof = energies(rng)
        build_prof(src_d, prof)
        src_vals = {}
        for li in range(L):
            p = f"model.layers.{li}."
            ent = []
            for nm, rows, cols in (("mlp.gate_proj.weight", I, H),
                                   ("mlp.up_proj.weight", I, H),
                                   ("mlp.down_proj.weight", H, I)):
                v = rng.integers(-8, 8, size=(rows, cols)).astype(np.int8)
                src_vals[p + nm] = v
                ent.append((p + nm, "U8", [rows * cols // 2],
                            hot_sort.pack_i4(v.reshape(-1)).tobytes()))
                qs = (rng.random(rows) * 0.1).astype(np.float32)   # per-row scales
                if nm == "mlp.up_proj.weight" and li == 0:
                    qs = (rng.random(I * 2) * 0.1).astype(np.float32)   # wrong width on purpose
                src_vals[p + nm + ".qs"] = qs
                ent.append((p + nm + ".qs", "F32", [qs.size], qs.tobytes()))
            save_st(os.path.join(src_d, f"model-{li:05d}.safetensors"), ent)
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("i4 sort rc=0", rc == 0, msg[-400:])
        check("group-wise-width qs warns (left unsorted, not crashed)",
              "scale elements" in msg and "UNSORTED" in msg, msg[-400:])
        ok = True
        for li in range(L):
            order = order_of(prof, li)
            got = read_st_raw(f"{out}/model-{li:05d}.safetensors")
            p = f"model.layers.{li}."
            for nm, rows, cols in (("mlp.gate_proj.weight", I, H),
                                   ("mlp.up_proj.weight", I, H),
                                   ("mlp.down_proj.weight", H, I)):
                e = hot_sort.unpack_i4(np.frombuffer(got[p + nm][0], np.uint8)).reshape(rows, cols)
                want = (src_vals[p + nm][order] if "down" not in nm
                        else src_vals[p + nm][:, order])
                ok &= np.array_equal(e, want)
                q = np.frombuffer(got[p + nm + ".qs"][0], np.float32)
                wq = src_vals[p + nm + ".qs"]
                if wq.size == I and "down" not in nm:
                    ok &= np.array_equal(q, wq[order])
                else:
                    ok &= np.array_equal(q, wq)      # wrong-width/down qs: untouched
        check("i4 nibbles round-trip through the permutation; per-row qs permuted", ok)


def test_bad_span():
    print("== corrupt header: F16 span != shape*numel -> loud refusal")
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)
        base_files(src_d)
        build_prof(src_d, energies(rng))
        m = f16_matrix(rng, I, H)
        save_st(os.path.join(src_d, "model-00000.safetensors"),
                [(f"model.layers.0.mlp.gate_proj.weight", "F16", [I, H],
                  m.tobytes() + b"\x00" * (I * H * 2))])   # span 2x the shape
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("bad-span rc=2", rc == 2, f"rc={rc}")
        check("bad-span message names tensor + st_dump", "gate_proj.weight" in msg
              and "st_dump" in msg and "span" in msg, msg[-300:])
        rc2, msg2 = run(ST_DUMP, src_d)
        check("st_dump flags bad span (rc=1)", rc2 == 1 and "!!" in msg2, msg2[-300:])


def test_fused_geometry():
    print("== fused (2*shared_inter+1)*hidden F16 gate -> refusal WITH the fused hint")
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)
        base_files(src_d)
        build_prof(src_d, energies(rng))
        m = f16_matrix(rng, 2 * I + 1, H)
        save_st(os.path.join(src_d, "model-00000.safetensors"),
                [("model.layers.0.mlp.gate_proj.weight", "F16", [2 * I + 1, H], m.tobytes())])
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("fused rc=2", rc == 2, f"rc={rc}")
        check("fused hint present", "(2*shared_inter+1)*hidden" in msg and "fused" in msg,
              msg[-400:])


def test_duplicates():
    print("== cross-shard duplicate names: no index -> refusal; index -> canonical copy only")
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)
        base_files(src_d)
        prof = energies(rng)
        build_prof(src_d, prof)
        g0 = f16_matrix(rng, I, H)
        g1 = f16_matrix(rng, I, H)
        save_st(os.path.join(src_d, "model-00000.safetensors"),
                [("model.layers.0.mlp.gate_proj.weight", "F16", [I, H], g0.tobytes())])
        save_st(os.path.join(src_d, "model-00001.safetensors"),
                [("model.layers.0.mlp.gate_proj.weight", "F16", [I, H], g1.tobytes())])
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("dup-no-index rc=2", rc == 2, f"rc={rc}")
        check("dup message names both shards + st_init rule",
              "model-00000.safetensors" in msg and "model-00001.safetensors" in msg
              and "st_init" in msg, msg[-400:])
        rc_d, msg_d = run(ST_DUMP, src_d)
        check("st_dump flags unresolved duplicate (rc=1)", rc_d == 1 and "REFUSES" in msg_d,
              msg_d[-300:])
        # now add the index: shard 1 is authoritative
        json.dump({"weight_map": {"model.layers.0.mlp.gate_proj.weight": "model-00001.safetensors"}},
                  open(os.path.join(src_d, "model.safetensors.index.json"), "w"))
        rc2, msg2 = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("dup+index rc=0", rc2 == 0, msg2[-400:])
        got0 = read_st_raw(f"{out}/model-00000.safetensors")
        got1 = read_st_raw(f"{out}/model-00001.safetensors")
        order = order_of(prof, 0)
        ok = ("model.layers.0.mlp.gate_proj.weight" not in got0 and
              np.array_equal(np.frombuffer(got1["model.layers.0.mlp.gate_proj.weight"][0],
                                           np.float16).reshape(I, H), g1[order]))
        check("ignored copy dropped; canonical copy permuted", ok)
        rc_d2, msg_d2 = run(ST_DUMP, src_d)
        check("st_dump explains index resolution (rc=0)", rc_d2 == 0 and "ignores the other" in msg_d2,
              msg_d2[-300:])


def test_bf16_passthrough():
    print("== BF16 tensor: passed through verbatim, sort still succeeds")
    rng = np.random.default_rng(6)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)

        def extra(li):
            b = rng.integers(0, 256, size=H * 2, dtype=np.uint8).tobytes()
            return [(f"model.layers.{li}.linear_attn.norm.weight", "BF16", [H], b)]
        src = build_f16(src_d, rng, extra=extra)
        build_prof(src_d, energies(rng))
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("bf16 sort rc=0", rc == 0, msg[-400:])
        ok = True
        for li in range(L):
            got = read_st_raw(f"{out}/model-{li:05d}.safetensors")
            raw = read_st_raw(f"{src_d}/model-{li:05d}.safetensors")
            k = f"model.layers.{li}.linear_attn.norm.weight"
            ok &= got[k] == raw[k]
        check("bf16 bytes+shape preserved verbatim", ok)


def test_noop_warning():
    print("== shared_expert naming only: sort succeeds but WARNS (no silent no-op)")
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)
        base_files(src_d)
        build_prof(src_d, energies(rng))
        m = f16_matrix(rng, I, H)
        save_st(os.path.join(src_d, "model-00000.safetensors"),
                [("model.layers.0.mlp.shared_expert.gate_proj.weight", "F16", [I, H],
                  m.tobytes())])
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("noop rc=0", rc == 0, msg[-400:])
        check("noop warns loudly", "WARNING" in msg and "UNSORTED" in msg, msg[-400:])


def test_metadata_preserved():
    print("== __metadata__ (colibri.fmt stamps) survives the sort")
    rng = np.random.default_rng(8)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)
        base_files(src_d)
        build_prof(src_d, energies(rng))
        m = f16_matrix(rng, I, H)
        save_st(os.path.join(src_d, "model-00000.safetensors"),
                [("model.layers.0.mlp.gate_proj.weight", "F16", [I, H], m.tobytes())],
                metadata={"colibri.fmt": "test=1"})
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        got = read_st_raw(f"{out}/model-00000.safetensors")
        check("metadata rc=0 and preserved", rc == 0 and
              got.get("__metadata__") == {"colibri.fmt": "test=1"}, msg[-300:])


def test_offsets_regression():
    """THE q27 crash, in miniature. v2 read data_offsets as (offset, LENGTH); they
    are (begin, END). A tensor at begin>0 came back with begin extra bytes. On the
    real shard the F16 gate_proj sat behind one (I,H) F16 matrix (2*I*H bytes) and
    one (H,) F16 vector (2*H bytes), so v2's array had
        (begin + span)/2 = (2*I*H + 2*H + 2*I*H)/2 = (2*I + 1) * H
    elements — with I=17408, H=5120 exactly the reported 178,263,040. This fixture
    reproduces that layout at (I=8, H=16) and asserts the permutation is right."""
    print("== regression: [begin,end) offsets with predecessor tensors (the q27 crash)")
    rng = np.random.default_rng(9)
    with tempfile.TemporaryDirectory() as td:
        src_d, out = f"{td}/c", f"{td}/o"
        os.makedirs(src_d)
        base_files(src_d)
        prof = energies(rng)
        build_prof(src_d, prof)
        pre = f16_matrix(rng, I, H)                       # one (I,H) F16 predecessor
        vec = f16_matrix(rng, 1, H)                       # one (H,) F16 vector
        gate = f16_matrix(rng, I, H)
        up = f16_matrix(rng, I, H)
        down = f16_matrix(rng, H, I)
        p = "model.layers.0."
        save_st(os.path.join(src_d, "model-00000.safetensors"), [
            (p + "linear_attn.in_proj_ba.weight", "F16", [I, H], pre.tobytes()),
            (p + "input_layernorm.weight", "F16", [H], vec.tobytes()),
            (p + "mlp.gate_proj.weight", "F16", [I, H], gate.tobytes()),
            (p + "mlp.up_proj.weight", "F16", [I, H], up.tobytes()),
            (p + "mlp.down_proj.weight", "F16", [H, I], down.tobytes()),
        ])
        # document the crash arithmetic: under v2 semantics the gate array would be
        begin = 2 * I * H + 2 * H
        v2_size = (begin + 2 * I * H) // 2
        check("v2 arithmetic reproduces (2I+1)*H elements", v2_size == (2 * I + 1) * H,
              f"{v2_size} != {(2 * I + 1) * H}")
        rc, msg = run(HOT_SORT, src_d, f"{src_d}/prof.json", out)
        check("real-layout sort rc=0", rc == 0, msg[-400:])
        order = order_of(prof, 0)
        got = read_st_raw(f"{out}/model-00000.safetensors")
        ok = (np.array_equal(np.frombuffer(got[p + "mlp.gate_proj.weight"][0], np.float16).reshape(I, H), gate[order]) and
              np.array_equal(np.frombuffer(got[p + "mlp.up_proj.weight"][0], np.float16).reshape(I, H), up[order]) and
              np.array_equal(np.frombuffer(got[p + "mlp.down_proj.weight"][0], np.float16).reshape(H, I), down[:, order]) and
              np.array_equal(np.frombuffer(got[p + "linear_attn.in_proj_ba.weight"][0], np.float16).reshape(I, H), pre) and
              np.array_equal(np.frombuffer(got[p + "input_layernorm.weight"][0], np.float16).reshape(1, H), vec))
        check("gate/up/down permuted; predecessors passed through byte-exact", ok)


def main():
    for t in (test_f16_clean, test_i4_clean, test_offsets_regression, test_bad_span,
              test_fused_geometry, test_duplicates, test_bf16_passthrough,
              test_noop_warning, test_metadata_preserved):
        t()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("\nall hot_sort/st_dump fixture gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
