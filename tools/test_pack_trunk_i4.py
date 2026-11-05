#!/usr/bin/env python3
"""test_pack_trunk_i4.py — offline gates for the dense-trunk int4 packer.

No engine, no model weights, ~30 MB of scratch. Run: python3 tools/test_pack_trunk_i4.py

Gates
  G1  layout round-trip: dequantizing the packed bytes in THIS file's own convention
      reproduces the --emit-dequant oracle container exactly, and the oracle is
      within the 4-bit bound of the F16 source (per class).
  G2  cross-tool equivalence: pack(--profile) then hot_sort == hot_sort then
      pack(--profile), byte-for-byte, for the MLP tensors and their .qs. This is
      the real proof that the packer's nibble order, scale placement AND hotness
      permutation agree with the shipped sorter instead of merely coexisting.
  G3  predicates: every packed tensor is U8 with nbytes == I*O/2 and has an F32
      .qs of numel == O (the engine's load_tq -> matmul_map_i4 dispatch condition).
  G4  refusals: an already-int4 container is refused (never a silent double-pack);
      --components narrows the target set; a profile from another geometry is
      skipped with a warning instead of mis-permuting a layer.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hot_sort                                        # noqa: E402  (sibling tool, by design)
PACKER = os.path.join(HERE, "pack_trunk_i4.py")
GEN = os.path.join(ROOT, "scripts", "gen_small_dense_f16.py")

FAILS = []

def check(ok, label, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)

def st_index(d):
    """{name: (path, entry, dstart)} for a container dir."""
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, e in hdr.items():
            if k != "__metadata__":
                out[k] = (p, e, 8 + n)
    return out

def read_tensor(idx, name):
    p, e, dst = idx[name]
    o0, o1 = e["data_offsets"]
    with open(p, "rb") as f:
        f.seek(dst + o0)
        raw = f.read(o1 - o0)
    dt = {"U8": np.uint8, "I8": np.int8, "F16": np.float16,
          "F32": np.float32}[e["dtype"]]
    return np.frombuffer(raw, dtype=dt), e

def dequant_u8(raw, scales, I):
    """This file's independent re-implementation of the engine's reader: byte k ->
    element 2k low nibble, 2k+1 high nibble, two's complement, times scale[row]."""
    b = raw.astype(np.int16)
    lo, hi = b & 0xF, b >> 4
    q = np.empty(b.size * 2, dtype=np.int16)
    q[0::2] = np.where(lo & 8, lo - 16, lo)
    q[1::2] = np.where(hi & 8, hi - 16, hi)
    return q.reshape(-1, I).astype(np.float32) * scales.reshape(-1, 1)

def run(*args, expect_rc: "int | None" = 0, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=e)
    if expect_rc is not None and p.returncode != expect_rc:
        print(f"    rc={p.returncode} (wanted {expect_rc}): {' '.join(args)}")
        print("    stderr tail:", "\n    ".join(p.stderr.strip().splitlines()[-4:]))
        print("    stdout tail:", "\n    ".join(p.stdout.strip().splitlines()[-4:]))
    return p

def main():
    tmp = tempfile.mkdtemp(prefix="pack_i4_test_")
    ok = True
    try:
        src = os.path.join(tmp, "src")
        print("== build an F16 dense container (real heads, field dialect) ==")
        p = run(GEN, env={"SMALL_OUT": src})
        check(p.returncode == 0 and os.path.isdir(src), "G0 generator ran", src)
        idx = st_index(src)
        f16 = {n: e for n, (_, e, _) in idx.items() if e["dtype"] == "F16"}
        packed_dir = os.path.join(tmp, "packed")
        deq_dir = os.path.join(tmp, "deq")

        print("\n== G1/G3: pack with the dequant oracle, then verify ==")
        p = run(PACKER, src, packed_dir, "--verify", "--emit-dequant", deq_dir)
        check(p.returncode == 0, "packer exit 0")
        check("verify OK" in p.stdout, "packer --verify reported OK")
        pidx, didx = st_index(packed_dir), st_index(deq_dir)
        man = json.load(open(os.path.join(packed_dir, "trunk_i4_manifest.json")))
        n_t = man["totals"]["tensors"]
        check(n_t > 40, f"packed {n_t} dense tensors", f"classes={man['components']}")
        check(man["totals"]["out_bytes"] * 3.5 < man["totals"]["src_bytes"],
              "int4 output is >=3.5x smaller",
              f"{man['totals']['src_gb']} -> {man['totals']['out_gb']} GB")

        worst = {}
        predfail, deqfail = [], []
        for t in man["tensors"]:
            name, I, O = t["name"], t["I"], t["O"]
            raw, e = read_tensor(pidx, name)
            qs, qe = read_tensor(pidx, name + ".qs")
            ref, _ = read_tensor(didx, name)
            if e["dtype"] != "U8" or raw.size != I * O // 2:
                predfail.append(f"{name}: {e['dtype']} nbytes {raw.size} != {I * O // 2}")
                continue
            if qe["dtype"] != "F32" or qs.size != O:
                predfail.append(f"{name}.qs: {qe['dtype']} numel {qs.size} != {O}")
                continue
            mine = dequant_u8(raw, qs, I).astype(np.float16)
            if not np.array_equal(mine, ref.reshape(O, I).astype(np.float16)):
                deqfail.append(name)
            worst[t["cls"]] = max(worst.get(t["cls"], 0.0), t["rel_l2_err"])
        check(not predfail, "G3 every target is U8 + nbytes==I*O/2 + F32 .qs numel==O",
              f"{len(predfail)} problem(s)" + (f": {predfail[:2]}" if predfail else ""))
        check(not deqfail, "G1 packed bytes dequantize to the oracle container exactly",
              f"{len(deqfail)} differ: {deqfail[:3]}")
        for cls, rel in sorted(worst.items()):
            check(rel < 0.25, f"G1 {cls}: 4-bit rel L2 vs F16 source {rel:.3f} < 0.25 "
                              f"(random Gaussian weights = worst case)")
        # passthrough integrity, spot-checked the same way --verify does it
        mism = [n for n, e in f16.items()
                if not n.endswith(".qs") and n in pidx and e["dtype"] in ("F16", "F32")
                and n not in man_names(man) and bytes(read_tensor(idx, n)[0]) !=
                bytes(read_tensor(pidx, n)[0])]
        check(not mism, "G1 passthrough F16 tensors byte-identical to source",
              f"{len(mism)} mismatch(es): {mism[:2]}")

        print("\n== G2: cross-tool equivalence with the shipped hot_sort ==")
        H, I = geo_of(src)
        prof = {"L%d" % li: {"energy": list(np.random.default_rng(li).random(I))}
                for li in sorted({int(n.split(".")[2]) for n in f16
                                  if n.startswith("model.layers.")})}
        ppath = os.path.join(tmp, "prof.json")
        json.dump(prof, open(ppath, "w"))
        # route A: pack (sorted in the same pass) -> hot_sort
        a1, a2 = os.path.join(tmp, "A_pack"), os.path.join(tmp, "A_sort")
        check(run(PACKER, src, a1, "--profile", ppath).returncode == 0, "route A: pack --profile")
        check(run(os.path.join(HERE, "hot_sort.py"), a1, ppath, a2).returncode == 0,
              "route A: hot_sort on the packed output")
        # route B: hot_sort the F16 container -> pack --profile
        b1, b2 = os.path.join(tmp, "B_sort"), os.path.join(tmp, "B_pack")
        check(run(os.path.join(HERE, "hot_sort.py"), src, ppath, b1).returncode == 0,
              "route B: hot_sort on the F16 container")
        check(run(PACKER, b1, b2, "--profile", ppath).returncode == 0, "route B: pack --profile")
        ai, bi = st_index(a2), st_index(b2)
        diff, same_n = [], 0
        for t in man["tensors"]:
            n = t["name"]
            for k in (n, n + ".qs"):
                if k in ai and k in bi:
                    if bytes(read_tensor(ai, k)[0]) == bytes(read_tensor(bi, k)[0]):
                        same_n += 1
                    else:
                        diff.append(k)
        check(not diff, f"G2 pack∘sort == sort∘pack byte-for-byte ({same_n} tensors compared)",
              f"{len(diff)} differ: {diff[:3]}")

        print("\n== G4: refusals and narrowing ==")
        p = run(PACKER, packed_dir, os.path.join(tmp, "again"), expect_rc=None)
        check(p.returncode != 0 and "already" in (p.stderr + p.stdout),
              "already-int4 input refused (no silent double-pack)",
              (p.stderr + p.stdout).strip().splitlines()[-1][:120])
        p = run(PACKER, src, os.path.join(tmp, "lm_headonly"),
                "--components", "lmhead", "--verify")
        m2 = json.load(open(os.path.join(tmp, "lm_headonly", "trunk_i4_manifest.json")))
        check(p.returncode == 0 and m2["totals"]["tensors"] == 1
              and m2["components"] == ["lmhead"], "--components lmhead packs exactly 1 tensor")
        bad = {"L0": {"energy": [0.1, 0.2]}}          # width 2 != mlp width
        bp = os.path.join(tmp, "badprof.json")
        json.dump(bad, open(bp, "w"))
        p = run(PACKER, src, os.path.join(tmp, "badprof"), "--profile", bp, expect_rc=None)
        both = p.stderr + p.stdout
        check(p.returncode != 0 and not os.path.exists(os.path.join(tmp, "badprof")),
              "profile from another geometry is a REFUSAL, not a silently-unsorted container",
              both.strip().splitlines()[-1][:110] if both.strip() else "")
        # a partial profile must warn about every layer it does not cover
        prof["L0"] = {"energy": [0.1]}
        json.dump(prof, open(bp, "w"))
        p = run(PACKER, src, os.path.join(tmp, "partprof"), "--profile", bp)
        check(p.returncode == 0 and "WARNING" in p.stderr and "UNSORTED" in p.stderr,
              "partial profile warns that uncovered layers serve UNSORTED",
              next((l for l in p.stderr.splitlines() if "WARNING" in l), "")[:110])
        p = run(PACKER, src, os.path.join(tmp, "d"), "--components", "nonsense")
        check(p.returncode != 0, "unknown --components value refused")
        p = run(PACKER, os.path.join(tmp, "empty"), os.path.join(tmp, "e"), expect_rc=None)
        check(p.returncode != 0, "non-container directory refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\nscratch {tmp} removed")
    print(f"\n{'ALL GATES PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


def man_names(man):
    return {t["name"] for t in man["tensors"]}


def geo_of(src):
    """(mlp_inter, ) read from the container's own meta — the test must not hardcode
    the generator's geometry (a generator change would silently invalidate the gate)."""
    meta = json.load(open(os.path.join(src, "qwen36_meta.json")))
    return meta["hidden"], (meta.get("shared_inter") or meta.get("moe_inter") or 0)


if __name__ == "__main__":
    sys.exit(main())
