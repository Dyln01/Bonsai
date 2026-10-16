
import numpy as np
rng = np.random.default_rng(11)
L, E = 48, 128
lines = [f"-1 {L} {E}", "-2 1 qwen36"]
for li in range(L):
    # Zipf over experts: rank 1..E, counts ~ 1/rank^1.0, normalized to ~50k per layer
    ranks = np.arange(1, E+1)
    w = 1.0 / ranks
    w = w / w.sum()
    # round to counts
    counts = np.rint(w * 50000).astype(int)
    # random permutation of expert ids per layer (hot set differs per layer)
    perm = rng.permutation(E)
    for e, c in zip(perm, counts):
        if c > 0:
            lines.append(f"{li} {e} {c}")
open("/home/dylan/synth_models/synth_coder30/.coli_usage", "w").write("\n".join(lines))
print("seeded .coli_usage:", len(lines), "records")
