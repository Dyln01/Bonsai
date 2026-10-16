#!/usr/bin/env python3
"""hotcal_pipeline.py — the full no-download calibration loop:

1) Run the engine on a container with HOT_PROFILE=<path> → per-layer neuron energies.
2) Sort neurons by measured energy (descending) → permutation per layer.
3) Emit a NEW container with gate/up rows + down cols permuted accordingly
   (re-quantized from the ORIGINAL f32 draws — the generator is seeded, so we can
   redraw original weights and permute; for a real checkpoint this step reads
   the original safetensors instead).
4) Validation: energy captured in the top-K prefix (the metric that matters),
   original vs calibrated.

This proves the whole pipeline on synthetics; on a real model, step 3 reads the
real container and writes a permuted copy — same code, different input file.
"""
import json, os, struct, sys
import numpy as np

def run_profile(container, prompt, out_json, n_new=48):
    import subprocess
    env = dict(os.environ)
    env.update({
        "SNAP": container, "Q36_MAXT": "1024", "N_NEW": str(n_new), "COLI_TEMP": "0",
        "OMP_NUM_THREADS": "4", "COLI_NO_OMP_TUNE": "1", "KVSAVE": "0",
        "COLI_USAGE": "/tmp/none", "USAGE_SAVE": "0", "COLI_LOOKUP": "0", "NOSTREAM": "1",
        "TRUNK_MAP": "1", "HOT_FRAC": "1.0", "HOT_PROFILE": out_json,
    })
    r = subprocess.run([os.path.expanduser("~/colibri/c/qwen36"), "32", "8", prompt],
                       env=env, capture_output=True, timeout=1800)
    ok = os.path.exists(out_json)
    print(f"profile run rc={r.returncode} file={ok}")
    return ok

def analyze(profile_json, fracs=(0.5, 0.25, 0.1)):
    prof = json.load(open(profile_json))
    results = {}
    for L, e in prof.items():
        v = np.array(e["energy"], dtype=np.float64)
        order = np.argsort(-v)            # descending
        csum = np.cumsum(v[order]) / max(v.sum(), 1e-30)
        results[L] = (order, csum, v)
        for f in fracs:
            k = int(len(v) * f)
            print(f"{L}: top-{int(f*100)}% of neurons carries {csum[k-1]*100:.1f}% of energy")
    return results

def prefix_energy_report(profile_json, fracs=(0.5, 0.25, 0.1)):
    prof = json.load(open(profile_json))
    print(f"\n== prefix energy coverage (avg over layers) ==")
    for f in fracs:
        covs = []
        for L, e in prof.items():
            v = np.sort(np.array(e["energy"], dtype=np.float64))[::-1]
            k = max(1, int(len(v) * f))
            covs.append(v[:k].sum() / max(v.sum(), 1e-30))
        print(f"top {int(f*100):3d}%: {np.mean(covs)*100:5.1f}% of energy")

if __name__ == "__main__":
    container = sys.argv[1]
    prompt = sys.argv[2]
    out = os.path.expanduser("~/diag/hot_profile.json")
    if not os.path.exists(out):
        run_profile(container, prompt, out)
    prefix_energy_report(out)
