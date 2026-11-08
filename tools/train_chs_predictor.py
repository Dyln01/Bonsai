#!/usr/bin/env python3
"""train_chs_predictor.py -- the offline half of Cascade-Hot Streaming.

CHS's premise (docs/CHS_QWEN36.md) is that the per-token active set churns only a
little, so a cheap predictor can prefetch just the delta. Its runtime half is an
engine change whose value depends entirely on two numbers that can be measured
OFFLINE from dumps the engine already writes:

  1. CHURN -- how much does the true per-token top-K set actually change between
     consecutive tokens? (--delta-report). No predictor can beat this bound.
  2. BEAT-STatic -- does a fit x -> top-K clear the pre-registered rule (>= +5 pp
     energy coverage over BOTH static-A and GRIFFIN)? If not, the predictor is a
     no-op and the engine-side prefetch would fetch nothing useful.

This tool measures both, and writes the predictor ONLY in the form the engine can
actually load: the natural ridge target is W[D, I] (5120x17408 = 356 MB per layer,
22.8 GB across 64 layers -- unloadable), so the map is factorized to
    score(x) = (x @ Vr) @ (Ur @ Sr).T          Vr[D, r], UrSr[I, r]
and the tool reports coverage AS A FUNCTION OF RANK, i.e. what the engine's rank
budget buys, instead of pretending the full map fits.

Dump format (what engine/qwen36.c + tools/patch_profile_tok.py actually write --
NOT a per-token .npz; the document's sketch globs files that do not exist):
    <prefix>_L<l>.bin, records of  D f32 (x) | K u32 (top-K neuron idx) | K f32 (their energies)
Record bytes = D*4 + K*4 + K*4. Both blocks after x MUST be read with one structured
dtype: reading the whole record as f32 turns the index bytes into denormals and every
index casts to 0 (a silent all-zeros predictor).

Usage:
  train_chs_predictor.py --dumps dumps/train --dumps-eval dumps/eval \\
      --D 5120 --I 17408 --K 64 --rank 64 -o MODEL/chs_predictor.npz
  train_chs_predictor.py --dumps dumps/train --delta-report      # premise check only
  train_chs_predictor.py --dumps A --rank-scan 16,32,64,128      # the rank/reward curve

Training is a DEV-BOX job: it materializes W[D, I] (356 MB per layer at 27B geometry) and
SVDs it. The field box only ever loads the artifact (A[D, r] + B[r, I] per layer).
"""
import argparse
import json
import os
import sys

import numpy as np

# Record layout on disk. Keep in sync with engine/qwen36.c's Q36_PROFILE_TOK writer.
def rec_dtype(D, K):
    return np.dtype([("x", np.float32, (D,)), ("idx", np.uint32, (K,)),
                     ("en", np.float32, (K,))])


def n_layers_of(prefix):
    n = 0
    while os.path.exists(f"{prefix}_L{n}.bin"):
        n += 1
    return n


def load(prefix, D, K, layers=None):
    dt = rec_dtype(D, K)
    n_layers = layers or n_layers_of(prefix)
    if n_layers == 0:
        die(f"no dumps at {prefix}_L*.bin -- run the engine with "
            f"Q36_PROFILE_TOK={prefix} first")
    xs, tks, es = [], [], []
    for l in range(n_layers):
        rec = np.fromfile(f"{prefix}_L{l}.bin", dtype=dt)
        if rec.size == 0:
            die(f"{prefix}_L{l}.bin is empty")
        assert rec.dtype.itemsize == D * 4 + K * 4 + K * 4
        xs.append(np.ascontiguousarray(rec["x"], dtype=np.float32))
        tks.append(np.ascontiguousarray(rec["idx"]))
        es.append(np.ascontiguousarray(rec["en"], dtype=np.float32))
    return xs, tks, es


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def sparse_target(idx, en, I):
    """Dense (n_tok, I) target: |energy| at the recorded indices, 0 elsewhere.

    |en| because everything downstream (coverage, the static baselines, the engine's own
    fabs-based profile) ranks by magnitude; a signed target would train the fit to rank by a
    quantity no arm is scored on.
    """
    y = np.zeros((idx.shape[0], I), dtype=np.float32)
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    y[rows, idx.reshape(-1)] = np.abs(en).reshape(-1)
    return y


def coverage(pred_idx, true_idx, true_en):
    """Mean fraction of a token's TRUE top-K |energy| that the predicted set covers.

    Two mistakes this function exists to avoid, both found by running it:

    1. Marking `tmap[true_idx] = True` on a 2-D array marks the UNION of every token's true
       set, so any prediction landing inside the union counts as a hit -- with 240 tokens x
       K=8 against I=96 the union is nearly the whole layer and every arm scores 1.0000
       (this is what the first version did, and what exp_predictor_q36.py still does). The
       mask must be built PER TOKEN.
    2. Summing SIGNED energies is not a fraction: ~31% of the recorded energies are
       negative, so a set that hits only negatives scores > 1.0 and appears to beat a
       perfect baseline. The metric uses |en|.

    Magnitude is also the engine-consistent choice: the corpus hot-profile accumulates
    `acc[i] += fabs(row[i])` (engine/qwen36.c), i.e. the shipped prefix is ranked by
    magnitude, not by the signed per-token value the dump records.
    """
    en = np.abs(true_en)
    nb, K = true_idx.shape
    nidx = int(max(true_idx.max(), pred_idx.max())) + 1
    lut = np.zeros((nb, nidx), dtype=np.float32)          # per token: |energy| at true idx
    lut[np.arange(nb)[:, None], true_idx.astype(np.intp)] = en
    # pred indices come from argsort, so they are distinct within a token: no dedupe needed.
    hit = np.take_along_axis(lut, pred_idx.astype(np.intp), axis=1).sum(axis=1)
    tot = en.sum(axis=1)
    return float((hit / np.maximum(tot, 1e-30)).mean())


def churn_report(tks, K, I, D, layers):
    """CHS premise: how much the true per-token top-K set moves between tokens, in neurons
    and in the bytes those neurons would cost to fetch (int4, one neuron = gate+up+down
    rows of width D)."""
    print("== CHS premise: per-token churn of the TRUE top-K set ==")
    print(f"{'layer':>6} {'identical':>10} {'mean overlap':>13} {'delta neurons':>14}")
    tot_same = tot = 0
    deltas = []
    for l in range(layers):
        t = tks[l]
        if t.shape[0] < 2:
            continue
        same = int(sum(1 for i in range(t.shape[0] - 1) if np.array_equal(t[i], t[i + 1])))
        ov = np.mean([len(set(t[i].tolist()) & set(t[i + 1].tolist()))
                      for i in range(t.shape[0] - 1)])
        tot_same += same
        tot += t.shape[0] - 1
        deltas.append(K - ov)
        print(f"{l:>6} {100 * same / (t.shape[0] - 1):>9.1f}% {ov:>7.1f}/{K:<5} {K - ov:>14.1f}")
    if not deltas:
        return None
    d_n = float(np.mean(deltas))
    per_neuron_B = 3 * D // 2                     # int4: 3 rows (gate/up/down) x D/2 bytes
    per_tok_MB = d_n * layers * per_neuron_B / 1e6
    print(f"\nmean delta {d_n:.1f} of {K} neurons/layer = {100 * d_n / K:.1f}% of the hot set")
    print(f"=> delta traffic at int4 (3 rows x D/2 B/neuron, {layers} layers): "
          f"{per_tok_MB:.1f} MB/token")
    print(f"   identical consecutive sets: {tot_same}/{tot} "
          f"({100 * tot_same / max(tot, 1):.1f}%) -- a predictor cannot beat 0 churn, and "
          f"cannot help when churn is already ~0.")
    return dict(delta_neurons=d_n, delta_frac=d_n / K, delta_mb_per_token=per_tok_MB,
                identical_frac=tot_same / max(tot, 1))


def fit(prefix, D, I, K, layers, holdout=0.4):
    """Ridge fit on a holdout split, plus the static-A and GRIFFIN baselines the rule needs.
    `holdout` is a SAME-CORPUS split: the pre-registered G2 rule wants a disjoint corpus
    (pass --dumps-eval for that), so a holdout-only run is labelled as such."""
    Xs, Tks, Es = load(prefix, D, K, layers)
    n = Xs[0].shape[0]
    cut = int(n * (1 - holdout))
    print(f"dumps={prefix} layers={len(Xs)} tokens={n} "
          f"(train[0:{cut}], holdout[{cut}:{n}]) D={D} I={I} K={K}")
    return Xs, Tks, Es, cut


def ridge(Xa, Ya, D, lam=1.0):
    """W = (X^T X + lam I)^-1 X^T Y -- the repo's proven fit (scripts/exp_predictor_q36.py)."""
    XtX = Xa.T @ Xa + lam * np.eye(D, dtype=np.float32)
    return np.linalg.solve(XtX, Xa.T @ Ya)


def griffin_set(Yb, nb, K, SEQ=256, PR=128):
    """GRIFFIN v5 simulated: per-sequence top-K by that sequence's prompt-part aggregate."""
    n_seq = max(1, nb // SEQ)
    gsel = np.zeros((nb, K), dtype=np.uint32)
    for s in range(n_seq):
        pr = slice(s * SEQ, s * SEQ + PR)
        sel = np.argsort(-Yb[pr].sum(axis=0))[:K].astype(np.uint32)
        gsel[s * SEQ + PR:(s + 1) * SEQ] = sel
        if s == 0:                                  # sequences shorter than PR: fill from start
            gsel[:PR] = sel
    return gsel, n_seq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="dump prefix (train corpus)")
    ap.add_argument("--dumps-eval", help="disjoint dump prefix (the pre-registered G2 rule)")
    ap.add_argument("--D", type=int, required=True, help="hidden size (record x width)")
    ap.add_argument("--I", type=int, required=True, help="MLP intermediate (target width)")
    ap.add_argument("--K", type=int, default=64, help="top-K recorded in the dumps")
    ap.add_argument("--rank", type=int, default=64, help="factorization rank to write")
    ap.add_argument("--rank-scan", help="comma list of ranks; reports the rank/reward curve")
    ap.add_argument("--max-rank-mb", type=float, default=8.0,
                    help="per-layer RAM budget for the artifact at the target geometry")
    ap.add_argument("--holdout", type=float, default=0.4)
    ap.add_argument("--layers", type=int, help="layer count (default: count the dump files)")
    ap.add_argument("--delta-report", action="store_true", help="CHS premise check only")
    ap.add_argument("-o", "--out", help="predictor artifact (npz)")
    ap.add_argument("--json", help="write the report JSON here")
    a = ap.parse_args()

    layers = a.layers or n_layers_of(a.dumps)
    if a.delta_report:
        _, T, _ = load(a.dumps, a.D, a.K, layers)
        rep = churn_report(T, a.K, a.I, a.D, layers)
        if a.out is None:
            return
        if a.json:
            json.dump(rep, open(a.json, "w"), indent=1)

    Xs, Tks, Es, cut = fit(a.dumps, a.D, a.I, a.K, layers, a.holdout)
    if a.dumps_eval:
        Xe, Te, Ee = load(a.dumps_eval, a.D, a.K, layers)
        which = "disjoint corpus (pre-registered G2)"
    else:
        Xe = [x[cut:] for x in Xs]
        Te = [t[cut:] for t in Tks]
        Ee = [e[cut:] for e in Es]
        which = ("SAME-CORPUS HOLDOUT -- NOT the pre-registered disjoint-corpus gate "
                 "(pass --dumps-eval to get one)")
    print(f"eval: {which}")

    ranks = ([int(r) for r in a.rank_scan.split(",")] if a.rank_scan else [a.rank])
    I, D, K = a.I, a.D, a.K
    art = {}
    table = {r: {"pred": [], "static": [], "griffin": []} for r in ranks}
    for l in range(layers):
        Ya = sparse_target(Tks[l][:cut], Es[l][:cut], I)
        W = ridge(Xs[l][:cut], Ya, D)                     # [D, I]
        Yb = sparse_target(Te[l], Ee[l], I)
        nb = Xe[l].shape[0]
        staticA = np.argsort(-Ya.sum(axis=0))[:K].astype(np.uint32)
        gsel, _ = griffin_set(Yb, nb, K)
        # W is [D, I]: score(x) = x @ W. Factorize W ~ A[d, r] @ B[r, I] so the engine holds
        # two small matrices per layer and scores in the target space (never the 356 MB W).
        U, S, Vt = np.linalg.svd(W, full_matrices=False)   # U[D,k] S[k] Vt[k,I], k=min(D,I)
        for r in ranks:
            A = U[:, :r].astype(np.float32)                # [D, r]
            B = (S[:r, None] * Vt[:r]).astype(np.float32)  # [r, I]
            pred = np.argsort(-((Xe[l] @ A) @ B), axis=1)[:, :K].astype(np.uint32)
            table[r]["pred"].append(coverage(pred, Te[l], Ee[l]))
            table[r]["static"].append(coverage(np.tile(staticA, (nb, 1)), Te[l], Ee[l]))
            table[r]["griffin"].append(coverage(gsel, Te[l], Ee[l]))
            if r == a.rank:
                art[f"L{l}_A"] = A                          # [D, r]
                art[f"L{l}_B"] = B                          # [r, I]
        print(f"L{l:2d} " + "  ".join(
            f"r{r}: pred={table[r]['pred'][-1]:.4f}/static={table[r]['static'][-1]:.4f}"
            for r in ranks))

    print("\n== mean energy coverage (rule: pred >= static AND >= griffin, each by +5 pp) ==")
    out = {}
    print(f"{'rank':>6} {'pred':>8} {'static-A':>9} {'griffin-j':>10} "
          f"{'vs static':>10} {'vs griffin':>11} {'verdict':>8} {'MB/layer':>9} {'MB total':>9}")
    for r in ranks:
        p = float(np.mean(table[r]["pred"]))
        s = float(np.mean(table[r]["static"]))
        g = float(np.mean(table[r]["griffin"]))
        mb = (I * r + D * r) * 4 / 1e6
        ok = (p - s >= 0.05) and (p - g >= 0.05)
        out[r] = dict(pred=p, static=s, griffin=g, verdict="PASS" if ok else "FAIL",
                      mb_per_layer=mb, mb_total=mb * layers)
        print(f"{r:>6} {p:>8.4f} {s:>9.4f} {g:>10.4f} {p - s:>+10.4f} {p - g:>+11.4f} "
              f"{'PASS' if ok else 'FAIL':>8} {mb:>9.2f} {mb * layers:>9.2f}")

    best = max(ranks, key=lambda r: out[r]["pred"] - max(out[r]["static"], out[r]["griffin"]))
    cleared = out[a.rank]["verdict"] == "PASS"
    print(f"\nrank budget: the written artifact is rank {a.rank} "
          f"({out[a.rank]['mb_per_layer']:.2f} MB/layer, {out[a.rank]['mb_total']:.1f} MB "
          f"across {layers} layers; --max-rank-mb {a.max_rank_mb:g}); best tested rank for the "
          f"rule is {best}")
    if not cleared:
        print(f"VERDICT: the predictor does NOT clear the pre-registered rule at rank {a.rank} "
              f"({out[a.rank]['pred'] - out[a.rank]['static']:+.4f} vs static-A, "
              f"{out[a.rank]['pred'] - out[a.rank]['griffin']:+.4f} vs griffin-j; both must be "
              f">= +0.0500). Per the contract the runtime prefetch is NOT unlocked, so the "
              f"artifact is written with gate_cleared=false and the engine must not prefetch "
              f"from it.")
    if a.out:
        art.update(gate_cleared=np.array([cleared]), rank=np.array([a.rank]),
                   D=np.array([D]), I=np.array([I]), K=np.array([K]),
                   layers=np.array([layers]), holdout=np.array([a.holdout]),
                   disjoint_corpus=np.array([bool(a.dumps_eval)]),
                   coverage_pred=np.array([out[a.rank]["pred"]]),
                   coverage_static=np.array([out[a.rank]["static"]]),
                   coverage_griffin=np.array([out[a.rank]["griffin"]]))
        np.savez_compressed(a.out, **art)
        print(f"wrote {a.out} (gate_cleared={cleared})")
    if a.json:
        json.dump(dict(which=which, ranks=out, layers=layers,
                       written_rank=a.rank, gate_cleared=cleared, best_rank=best),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
