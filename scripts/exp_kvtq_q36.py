#!/usr/bin/env python3
"""exp_kvtq_q36.py — KV_TQ battery for qwen36 (third KV tier: f32 -> KV8 -> KV_TQ).

Arms on the synthetic 27B-shaped container (~8k ctx, 48-token decode — the SAME
operating point as LOWRAM_QWEN36_FINAL arms A-D, so numbers are comparable):
  A_f32          f32 KV (engine defaults)        — baseline
  B_kv8          KV8=1 fp8 KV tier
  E_tq4_c1       KV_TQ=4 codec 1 (rotated int4, n/2 B/row)
  F_tq4_c0       KV_TQ=4 codec 0 (PolarQuant-4)
  G_tq3_c0       KV_TQ=3 codec 0 (PolarQuant-3, RAM-extreme)
  H_tq4c1_i4dense KV_TQ=4 codec 1 + COLI_DENSE_BITS=4 — lowest-RAM profile

Honest decode rates via COLI_TIMERS=1 (the 'Speed:' line divides by total wall
time including prefill — never report it as decode). Each arm appends to the
results JSON as it completes, so a timeout never loses finished arms.

Usage:
  exp_kvtq_q36.py [--snap DIR] [--maxt N] [--nnew N] [--out FILE] [--prompt FILE] arm ...
"""
import subprocess, os, re, json, sys

E       = os.path.expanduser("~/colibri/c/qwen36")
RE_RSS    = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_TTFT   = re.compile(r"TTFT: ([\d.]+) s")
RE_DTOK   = re.compile(r"\[timers\] decode: (\d+) tokens")
RE_TOTAL  = re.compile(r"\[timers\]\s+TOTAL\s+([\d.]+) ms\s+([\d.]+) ms/token", re.M)
RE_ATTN   = re.compile(r"\[timers\]\s+attention\s+([\d.]+) ms\s+([\d.]+) ms/token", re.M)
RE_ENC    = re.compile(r"\[enc\] prompt tokens: (\d+)")
RE_BANNER = re.compile(r"\[qwen36\] KV_TQ=\d+ codec=\d+ \([^)]*\): KV rows packed \d+ B[^\n]*")

ARMS = {
    "A_f32":            {},
    "B_kv8":            {"KV8": "1"},
    "E_tq4_c1":         {"KV_TQ": "4", "KV_TQ_CODEC": "1"},
    "F_tq4_c0":         {"KV_TQ": "4", "KV_TQ_CODEC": "0"},
    "G_tq3_c0":         {"KV_TQ": "3", "KV_TQ_CODEC": "0"},
    "H_tq4c1_i4dense":  {"KV_TQ": "4", "KV_TQ_CODEC": "1", "COLI_DENSE_BITS": "4"},
    "J_f32_flash":      {"Q36_FLASH": "1"},
    "I_tq4c1_flash":    {"KV_TQ": "4", "KV_TQ_CODEC": "1", "Q36_FLASH": "1"},
}
SCRUB = ("IOTH_", "SNAP", "CTX", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "PIN",
         "SERVE", "PROMPT", "Q36_", "KV8", "KV_TQ", "COLI_DENSE_BITS", "COLI_DENSE_INT4",
         "COLI_TIMERS", "N_NEW", "NGEN", "COLI_TEMP", "TRUNK_", "HOT_", "GRIFFIN")

def main():
    a = sys.argv[1:]
    snap = os.path.expanduser("~/synth_models/synth_qwen27")
    maxt, nnew = "8192", "48"
    out = os.path.expanduser("~/colibri-research/harness/results_kvtq_q36.json")
    prompt = os.path.expanduser("~/prompt_8k.txt")
    while a and a[0].startswith("--"):
        if a[0] == "--snap":   snap = os.path.expanduser(a[1]); a = a[2:]
        elif a[0] == "--maxt": maxt = a[1]; a = a[2:]
        elif a[0] == "--nnew": nnew = a[1]; a = a[2:]
        elif a[0] == "--out":  out = os.path.expanduser(a[1]); a = a[2:]
        elif a[0] == "--prompt": prompt = os.path.expanduser(a[1]); a = a[2:]
        else: sys.exit("unknown option " + a[0])
    if not os.path.exists(prompt):
        open(prompt, "w").write("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 280)
    arms = a or list(ARMS)
    for label in arms:
        e = dict(os.environ)
        for k in list(e):
            if k.startswith(SCRUB):
                del e[k]
        e.update({"SNAP": snap, "KVSAVE": "0", "COLI_USAGE": os.path.expanduser("~/no_usage_file"),
                  "USAGE_SAVE": "0", "Q36_MAXT": maxt, "N_NEW": nnew, "COLI_TEMP": "0",
                  "COLI_TIMERS": "1"})
        e.update(ARMS[label])
        p = subprocess.run(["/usr/bin/time", "-v", E, "32", "8", prompt], env=e, capture_output=True)
        err = p.stderr.decode(errors="replace")
        m = RE_RSS.search(err)
        r = {"peak_rss_mb": round(int(m.group(1)) / 1024, 1) if m else None}
        m = RE_TTFT.search(err);  r["ttft_s"] = float(m.group(1)) if m else None
        m = RE_DTOK.search(err);  r["decode_tokens"] = int(m.group(1)) if m else None
        m = RE_TOTAL.search(err); r["decode_ms_per_tok"] = float(m.group(2)) if m else None
        m = RE_ATTN.search(err);  r["attn_ms_per_tok"] = float(m.group(2)) if m else None
        m = RE_ENC.search(err);   r["prompt_tokens"] = int(m.group(1)) if m else None
        b = RE_BANNER.search(err); r["banner"] = b.group(0) if b else None
        r["decode_tok_s"] = round(1000.0 / r["decode_ms_per_tok"], 3) if r["decode_ms_per_tok"] else None
        r["rc"] = p.returncode
        results = {}
        if os.path.exists(out):
            try: results = json.load(open(out))
            except Exception: results = {}
        results[label] = r
        json.dump(results, open(out, "w"), indent=2)
        print(f"{label:16s}: rss={r['peak_rss_mb']}MB ttft={r['ttft_s']}s dec={r['decode_tok_s']} tok/s "
              f"attn={r['attn_ms_per_tok']}ms/tok rc={r['rc']}", flush=True)
        if r["banner"]: print(f"    {r['banner']}", flush=True)

if __name__ == "__main__":
    main()
