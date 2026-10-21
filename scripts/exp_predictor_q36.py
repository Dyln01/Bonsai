#!/usr/bin/env python3
"""exp_predictor_q36.py -- Phase 4 offline gate (G2): predictor vs baselines.

Implements the decision rule pre-registered in docs/PREDICTOR_QWEN36.md:
on a DISJOINT eval corpus, per token, energy coverage of the selected
top-K against the TRUE top-K energies:

  static-A   top-K by train-corpus aggregate   (the HOT_FRAC model)
  griffin-j  top-K by each eval SEQUENCE's own prompt-part aggregate
             (GRIFFIN v5 simulated: 16 sequences of 256 tokens, first 128
              tokens = the prompt, last 128 = the evaluated tokens)
  predictor  per-token top-K of a ridge fit  x -> true-energy-vector,
             trained ONLY on the train corpus (one linear map per layer)

Decision: predictor must beat BOTH static-A and griffin-j by >= 5 pp mean
energy coverage. Everything runs on the Q36_PROFILE_TOK dumps.

Usage: exp_predictor_q36.py <dumpA_prefix> <dumpB_prefix> <D> <I> <K>
      (also writes the per-layer ridge weights to <dumpA_prefix>_pred.npz)
"""
import json, os, sys
import numpy as np

A, B, D, I, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
REC = D + K + K  # floats per record

def load(prefix, n_layers):
    # record = x[D] f32 | K u32 idx | K f32 energy -- a STRUCTURED dtype, else
    # the u32 bytes read as f32 denormals and every index casts to zero
    dt = np.dtype([("x", np.float32, D), ("idx", np.uint32, K), ("en", np.float32, K)])
    xs, tks, es = [], [], []
    for l in range(n_layers):
        rec = np.fromfile(f"{prefix}_L{l}.bin", dtype=dt)
        assert rec.size > 0, (prefix, l)
        xs.append(np.ascontiguousarray(rec["x"], dtype=np.float32))
        tks.append(np.ascontiguousarray(rec["idx"]))
        es.append(np.ascontiguousarray(rec["en"], dtype=np.float32))
    return xs, tks, es

def topk_mask(idx, energy, n_tok):
    """dense (n_tok, I) energy vector: top-K energies at their indices"""
    y = np.zeros((n_tok, I), dtype=np.float32)
    rows = np.repeat(np.arange(n_tok), K)
    y[rows, idx.reshape(-1)] = energy.reshape(-1)
    return y

def coverage(pred_idx, true_idx, true_en):
    """mean fraction of true-top-K energy covered by the predicted set"""
    tmap = np.zeros(I, dtype=bool); tmap[true_idx] = True
    tot = true_en.sum(axis=1)
    hit = np.where(tmap[pred_idx], true_en, 0.0).sum(axis=1)
    return float((hit / np.maximum(tot, 1e-30)).mean())

def main():
    n_layers = 0
    while os.path.exists(f"{A}_L{n_layers}.bin"): n_layers += 1
    Xa, Ta, Ea = load(A, n_layers)
    Xb, Tb, Eb = load(B, n_layers)
    na, nb = Xa[0].shape[0], Xb[0].shape[0]
    print(f"layers={n_layers} train={na} eval={nb} D={D} I={I} K={K}")

    # sequences for the GRIFFIN simulation on the eval corpus
    SEQ, PR = 256, 128
    n_seq = nb // SEQ
    seq_of = np.arange(nb) // SEQ

    table = {s: [] for s in ("static-A", "static-B", "griffin-j", "predictor")}
    weights = {}
    for l in range(n_layers):
        Ya = topk_mask(Ta[l], Ea[l], na)
        # ridge: W = (X^T X + lam I)^-1 X^T Y   (deterministic, one solve)
        XtX = Xa[l].T @ Xa[l] + 1.0 * np.eye(D, dtype=np.float32)
        XtY = Xa[l].T @ Ya
        W = np.linalg.solve(XtX, XtY)
        weights[f"L{l}"] = W

        # baselines
        aggA = Ya.sum(axis=0)
        staticA = np.argsort(-aggA)[:K]
        Yb = topk_mask(Tb[l], Eb[l], nb)
        aggB = Yb.sum(axis=0)
        staticB = np.argsort(-aggB)[:K]

        # per-sequence GRIFFIN: aggregate over each sequence's prompt part
        grif = np.zeros((nb, K), dtype=np.uint32)
        for s in range(n_seq):
            pr = slice(s*SEQ, s*SEQ + PR)
            gsel = np.argsort(-Yb[pr].sum(axis=0))[:K]
            ev = slice(s*SEQ + PR, (s+1)*SEQ)
            grif[ev] = gsel

        pred_idx = np.argsort(-(Xb[l] @ W), axis=1)[:, :K].astype(np.uint32)

        ev = np.zeros(nb, dtype=bool); ev[:n_seq*SEQ] = True; ev[:0] = False
        mask = np.zeros(nb, dtype=bool)
        for s in range(n_seq): mask[s*SEQ+PR:(s+1)*SEQ] = True

        table["static-A"].append(coverage(np.tile(staticA, (mask.sum(),1)),
                                          Tb[l][mask], Eb[l][mask]))
        table["static-B"].append(coverage(np.tile(staticB, (mask.sum(),1)),
                                          Tb[l][mask], Eb[l][mask]))
        table["griffin-j"].append(coverage(grif[mask], Tb[l][mask], Eb[l][mask]))
        table["predictor"].append(coverage(pred_idx[mask], Tb[l][mask], Eb[l][mask]))
        print(f"L{l:2d}  staticA={table['static-A'][-1]:.4f}  staticB={table['static-B'][-1]:.4f}  "
              f"griffin={table['griffin-j'][-1]:.4f}  predictor={table['predictor'][-1]:.4f}")

    np.savez_compressed(f"{A}_pred.npz", **weights)
    means = {k: float(np.mean(v)) for k, v in table.items()}
    print("\n== mean energy coverage over", n_layers, "layers ==")
    for k in ("static-A", "static-B", "griffin-j", "predictor"):
        print(f"  {k:10s} {means[k]:.4f}")
    p = means["predictor"]
    verdict = ("PASS" if (p - means["static-A"] >= 0.05 and p - means["griffin-j"] >= 0.05)
               else "FAIL")
    print(f"\nG2 decision: predictor vs static-A {p-means['static-A']:+.4f}, "
          f"vs griffin-j {p-means['griffin-j']:+.4f} -> {verdict} "
          f"(rule: >= +0.05 both; runtime work only on PASS)")
    json.dump({"means": means, "verdict": verdict, "per_layer": {k: v for k, v in table.items()}},
              open(f"{A}_g2.json", "w"), indent=1)

if __name__ == "__main__":
    main()
