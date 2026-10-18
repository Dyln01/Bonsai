#!/usr/bin/env python3
"""exp_softmax_q36.py — Q36_FAST_SOFTMAX A/B on the engine (softmax + flash sites).

Two arms on synth_qwen27 (15,960-token prompt, 48 new tokens, f32 KV, flash on
by default): SM_off (knob unset = stock arithmetic) vs SM_on
(Q36_FAST_SOFTMAX=1). Scrapes attention ms/token, honest decode (COLI_TIMERS),
TTFT, peak RSS, and confirms the knob banner. Writes
~/colibri-research/harness/results_softmax_q36.json.
"""
import subprocess, os, re, json, sys

E = os.path.expanduser("~/colibri/c/qwen36")
SNAP = os.path.expanduser("~/synth_models/synth_qwen27")
OUT = os.path.expanduser("~/colibri-research/harness/results_softmax_q36.json")
PROMPT = os.path.expanduser("~/prompt_8k.txt")
SCRUB = ("IOTH_", "SNAP", "CTX", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "PIN",
         "SERVE", "PROMPT", "Q36_", "KV8", "KV_TQ", "COLI_DENSE_BITS", "COLI_DENSE_INT4",
         "COLI_TIMERS", "N_NEW", "NGEN", "COLI_TEMP", "TRUNK_", "HOT_", "GRIFFIN",
         "Q36_FAST_SOFTMAX")
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
RE_TTFT = re.compile(r"TTFT: ([\d.]+) s")
RE_ATTN = re.compile(r"\[timers\]\s+attention\s+([\d.]+) ms\s+([\d.]+) ms/token", re.M)
RE_TOTAL = re.compile(r"\[timers\]\s+TOTAL\s+([\d.]+) ms\s+([\d.]+) ms/token", re.M)
RE_BANNER = re.compile(r"\[qwen36\] Q36_FAST_SOFTMAX[^\n]*")

def run(label, extra):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(SCRUB):
            del e[k]
    e.update({"SNAP": SNAP, "KVSAVE": "0", "COLI_USAGE": os.path.expanduser("~/no_usage_file"),
              "USAGE_SAVE": "0", "Q36_MAXT": "8192", "N_NEW": "48", "COLI_TEMP": "0",
              "COLI_TIMERS": "1"})
    e.update(extra)
    p = subprocess.run(["/usr/bin/time", "-v", E, "32", "8", PROMPT], env=e, capture_output=True)
    err = p.stderr.decode(errors="replace")
    m = RE_RSS.search(err)
    r = {"peak_rss_mb": round(int(m.group(1)) / 1024, 1) if m else None}
    m = RE_TTFT.search(err);  r["ttft_s"] = float(m.group(1)) if m else None
    m = RE_ATTN.search(err);  r["attn_ms_per_tok"] = float(m.group(2)) if m else None
    m = RE_TOTAL.search(err); r["decode_ms_per_tok"] = float(m.group(2)) if m else None
    r["decode_tok_s"] = round(1000.0 / r["decode_ms_per_tok"], 3) if r["decode_ms_per_tok"] else None
    b = RE_BANNER.search(err); r["banner"] = b.group(0)[:100] if b else None
    r["rc"] = p.returncode
    res = {}
    if os.path.exists(OUT):
        try: res = json.load(open(OUT))
        except Exception: res = {}
    res[label] = r
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"{label:8s}: rss={r['peak_rss_mb']}MB ttft={r['ttft_s']}s attn={r['attn_ms_per_tok']}ms/tok "
          f"dec={r['decode_tok_s']} tok/s rc={r['rc']} banner={'yes' if r['banner'] else 'no'}", flush=True)

if __name__ == "__main__":
    for label, extra in [("SM_off", {}), ("SM_on", {"Q36_FAST_SOFTMAX": "1"})]:
        run(label, extra)
