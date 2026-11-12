#!/usr/bin/env python3
"""analyze_lrs.py — can low-rank + sparse beat int4 at EQUAL BYTES on these tensors?

The LRS proposal (W ~= U @ V.T + S, rank r, top-k |residual| entries kept) targets the
classes with no HOT_FRAC dial: DeltaNet, lm_head, attention. Its sketch prices the win
against F16 ("1.5-3x on the untouchable ~70% of bytes"), which is the wrong baseline: the
field container that motivated it is ALREADY int4 (0.5 B/element). So the question this tool
answers, per tensor and per class, is the only one that decides the lever:

    at the SAME number of bytes, is LRS's reconstruction error lower than int4's?

Both sides are measured, not assumed:

  int4   the repo's own quantizer (per-row amax/7, low nibble = even element): m*n/2 B
         + m*4 B of scales. This reproduces the packed containers the engine reads.
  LRS    (m+n)*r*2 B (f16 factors) + nnz*(4+2) B (u32 index + f16 value). At a given rank
         the sparse budget is whatever is LEFT of the int4 byte count, so the comparison is
         byte-for-byte and never flatters one side.

Reconstruction error is a MECHANICAL property, which is why it is measurable and gateable on
synthetic containers under this repo's rule (synthetics are mechanical oracles, not numeric
ones -- no ppl claim is made here). Random weights are the worst case for low rank: their
spectrum is near-flat (Marchenko-Pastur), so `--self-test` also builds a strongly low-rank
matrix where LRS MUST win -- an analyzer that can only say "no" proves nothing.

Ranks are estimated with a randomized SVD (numpy-only, no scipy): the exact top-r subspace is
not needed to answer a byte-matched error question, and a truncated SVD of a 5120x17408
matrix is minutes per tensor.

Usage:
  analyze_lrs.py <container> [--classes deltanet,lm_head,attn,mlp] [--rank-scan 8,16,32,64,128]
                 [--max-elems 40e6] [--json out.json]
  analyze_lrs.py --self-test
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pack_trunk_i4 as pk   # the repo's own reader/quantizer/classifier: reuse, not re-implement


def rand_svd(W, r, p=8, seed=0, iters=2):
    """Top-r singular triplets via randomized range finding (numpy only)."""
    m, n = W.shape
    r = int(min(r, min(m, n)))
    if r <= 0:
        return np.zeros((m, 0), np.float32), np.zeros((0,), np.float32), np.zeros((n, 0), np.float32)
    rs = np.random.default_rng(seed)
    Om = rs.standard_normal((n, min(r + p, n))).astype(np.float32)
    Y = W @ Om
    for _ in range(iters):                     # power iteration sharpens the range
        Y = W @ (W.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ W
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ Ub, S, Vt


def int4_roundtrip(W):
    """The repo's quantizer: per-row amax/7, low nibble = even element. Returns (W_hat, bytes)."""
    m, n = W.shape
    amax = np.abs(W).max(axis=1, keepdims=True)
    scale = np.maximum(amax / 7.0, 1e-8)
    code = np.rint(np.clip(W / scale, -7, 7)).astype(np.int8)
    return code.astype(np.float32) * scale, m * n // 2 + m * 4


def rel_l2(W, Wh):
    den = float(np.sqrt((W.astype(np.float64) ** 2).sum()))
    num = float(np.sqrt(((W - Wh).astype(np.float64) ** 2).sum()))
    return num / max(den, 1e-30)


def lrs_error(W, r, nnz, seed=0):
    """rel L2 of U@V.T + top-nnz residual, and the bytes that representation costs."""
    m, n = W.shape
    U, S, Vt = rand_svd(W, r, seed=seed)
    Ar = (U * S) @ Vt if r > 0 else np.zeros_like(W)
    R = W - Ar
    if nnz > 0:
        k = int(min(nnz, R.size))
        flat = np.abs(R).ravel()
        idx = np.argpartition(flat, -k)[-k:] if k < flat.size else np.arange(flat.size)
        Sres = np.zeros_like(R).ravel()
        Sres[idx] = R.ravel()[idx]
        Wh = Ar + Sres.reshape(R.shape)
    else:
        Wh = Ar
    bytes_ = (m + n) * r * 2 + nnz * 6
    return rel_l2(W, Wh), bytes_


def verdict_for(W, ranks, label, seed=0):
    e_i4, b_i4 = int4_roundtrip(W)
    err_i4 = rel_l2(W, e_i4)
    rows = []
    for r in ranks:
        low_bytes = (W.shape[0] + W.shape[1]) * r * 2
        nnz = max(0, int((b_i4 - low_bytes) // 6))
        err, used = lrs_error(W, r, nnz, seed=seed)
        rows.append(dict(rank=r, nnz=nnz, lrs_err=err, lrs_bytes=used,
                         ratio=err / max(err_i4, 1e-30)))
    assert rows, "no ranks requested"
    best = min(rows, key=lambda x: x["ratio"])
    print(f"\n{label}  [{W.shape[0]}x{W.shape[1]}]")
    print(f"  int4 baseline: {b_i4 / 1e6:8.2f} MB  rel L2 {err_i4:.4f}")
    print(f"  {'rank':>5} {'sparse nnz':>11} {'LRS MB':>8} {'rel L2':>9} {'x int4 err':>11}")
    for x in rows:
        flag = "  <- beats int4" if x["ratio"] < 1.0 else ""
        print(f"  {x['rank']:>5} {x['nnz']:>11} {x['lrs_bytes'] / 1e6:>8.2f} "
              f"{x['lrs_err']:>9.4f} {x['ratio']:>11.2f}{flag}")
    return dict(shape=list(W.shape), int4_bytes=b_i4, int4_err=err_i4, ranks=rows,
                best=best, label=label, lrs_wins=bool(best["ratio"] < 1.0))


def self_test():
    """Both verdicts must be reachable: a low-rank matrix where LRS wins, and a flat-spectrum
    matrix where it cannot. An analyzer that can only say 'no' is not measuring anything."""
    bad = 0

    def check(ok, what, detail=""):
        nonlocal bad
        print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
        if not ok:
            bad += 1

    rs = np.random.default_rng(3)
    m, n, truerank = 512, 1024, 16
    A = rs.standard_normal((m, truerank)).astype(np.float32)
    B = rs.standard_normal((truerank, n)).astype(np.float32)
    low = (A @ B) + 0.001 * rs.standard_normal((m, n)).astype(np.float32)
    flat = rs.standard_normal((m, n)).astype(np.float32)

    v_low = verdict_for(low, [16, 32, 64], "self-test: genuinely rank-16 (+0.1% noise)")
    v_flat = verdict_for(flat, [16, 32, 64], "self-test: iid gaussian (flat spectrum)")
    check(v_low["lrs_wins"], "LRS WINS on a low-rank matrix (the analyzer can say yes)",
          f"best ratio {v_low['best']['ratio']:.3f}")
    check(not v_flat["lrs_wins"], "LRS LOSES on a flat-spectrum matrix",
          f"best ratio {v_flat['best']['ratio']:.3f}")
    # byte accounting must be exact on both sides
    r = 32
    low_bytes = (m + n) * r * 2
    expect_nnz = max(0, (v_flat["int4_bytes"] - low_bytes) // 6)
    err, used = lrs_error(flat, r, expect_nnz)
    check(used == low_bytes + expect_nnz * 6,
          "LRS bytes == factors + nnz*6", f"{used} vs {low_bytes + expect_nnz * 6}")
    check(abs(used - v_flat["int4_bytes"]) <= 6,
          "at the byte-matched nnz the two representations cost the same",
          f"LRS {used} vs int4 {v_flat['int4_bytes']}")
    # error must fall as the sparse budget grows (a monotonicity the sweep relies on)
    e_a, _ = lrs_error(flat, 8, 10_000, seed=1)
    e_b, _ = lrs_error(flat, 8, 40_000, seed=1)
    check(e_b < e_a, "more residual entries -> lower error", f"{e_a:.4f} -> {e_b:.4f}")
    # and the int4 baseline must reproduce the repo's measured range on random weights
    check(0.05 < v_flat["int4_err"] < 0.25,
          "the int4 baseline reproduces the repo's measured rel L2 band (~0.12) on random weights",
          f"{v_flat['int4_err']:.4f}")
    print("\nANALYZE_LRS SELF-TEST: " + ("PASS" if not bad else f"{bad} FAILED"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("container", nargs="?")
    ap.add_argument("--classes", default="deltanet,lm_head,attn,mlp",
                    help="which classes to analyze (comma list, or 'all')")
    ap.add_argument("--rank-scan", default="8,16,32,64,128")
    ap.add_argument("--max-elems", type=float, default=40e6,
                    help="skip tensors larger than this (a truncated SVD of a 5120x17408 "
                         "matrix costs minutes); use --rows to analyse a row sample instead")
    ap.add_argument("--rows", type=int, default=0,
                    help="analyse only the first N rows (a sample: its spectrum is NOT the "
                         "full matrix's, and a row sample flatters low rank)")
    ap.add_argument("--max-tensors", type=int, default=6)
    ap.add_argument("--json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.container:
        return self_test()

    ranks = [int(x) for x in a.rank_scan.split(",")]
    want = None if a.classes == "all" else {c.strip() for c in a.classes.split(",")}
    files, headers, owner = pk.build_index(a.container)
    print(f"container: {a.container}\n{len(owner)} tensor(s), {len(files)} shard(s); "
          f"classes: {sorted(want) if want else 'all'}")
    print("note: only FLOAT tensors are analyzed. A U8 tensor is already the int4 "
          "representation, so quantizing it again would compare int4 with itself -- point "
          "this at the SOURCE container (the one before pack_trunk_i4).")

    results, skipped = [], []
    for name in sorted(owner):
        cl = pk.classify(name)
        if cl is None:
            continue
        role, cls, layer = cl
        if want and cls not in want and not (cls in ("dnproj", "dnout") and "deltanet" in want):
            continue
        shard = owner[name]
        hdr, dstart = headers[shard]
        ent = hdr[name]
        dtype, shape = ent["dtype"], ent["shape"]
        esz = pk.ESZ.get(dtype)
        if esz is None or dtype in ("U8", "I8"):
            skipped.append((name, f"{dtype} (already packed/int8 -- see the note above)"))
            continue
        if len(shape) != 2:
            skipped.append((name, f"shape {shape} (not a 2-D GEMV tensor)"))
            continue
        rows, cols = int(shape[0]), int(shape[1])
        if rows * cols > a.max_elems:
            skipped.append((name, f"{rows}x{cols} > --max-elems {a.max_elems:g} "
                                  f"(a truncated SVD of this costs minutes)"))
            continue
        take = min(rows, a.rows) if a.rows else rows
        with open(os.path.join(a.container, shard), "rb") as f:
            W = pk.read_rows(f, dstart + ent["data_offsets"][0], take, cols, esz)
        results.append(verdict_for(W, ranks,
                                   f"{cls:7} {role:7} {name}"
                                   + (f" (first {take} of {rows} rows -- a SAMPLE)" if take < rows
                                      else "")))
        if len(results) >= a.max_tensors:
            break

    for nm, why in skipped:
        print(f"\nSKIP {nm}: {why}")
    wins = sum(1 for r in results if r["lrs_wins"])
    if results:
        print(f"\n== verdict: LRS beats int4 at EQUAL BYTES on {wins}/{len(results)} "
              f"analyzed tensor(s) ==")
        for r in results:
            b = r["best"]
            print(f"  {r['label'][:52]:54} best rank {b['rank']:>4}  "
                  f"{'WINS' if r['lrs_wins'] else 'loses'}  (x{b['ratio']:.2f} int4's error)")
        print("\nReading: 'loses' means the low-rank+sparse form needs MORE bytes than int4 "
              "to reach int4's error on that tensor. It is a property of the spectrum, so the "
              "same tool on a real container (--rows for the big ones) is the number that "
              "decides the lever -- random weights are the worst case by construction.")
    if a.json:
        json.dump(dict(container=a.container, ranks=ranks, results=results,
                       skipped=[[str(x) for x in s] for s in skipped]), open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
