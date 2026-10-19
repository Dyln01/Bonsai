#!/usr/bin/env python3
"""exp_prefix_q36.py — KV-prefix reuse across agent turns, measured.

The engine's kv_prefix machinery (record-where-fed + swap-with-state in
serve_mux) has never been measured on q36. This driver speaks the raw serve
protocol (SUBMIT/ACCEPT/DATA/DONE) and drives AGENT-shaped traffic:
every turn resends the whole transcript (prompt + previously generated text)
and appends a tool-result-shaped tail — the exact pattern an agent runtime
produces.

Two traffic shapes, both honest about what a SYNTHETIC container can measure:

  append-only (max_tok=1, arms {mux,single}_{on,off}): the model's single
    sampled token is emitted but never fed back, so the recorded state is
    EXACTLY the resent prefix — every turn is an exact extension of the last.
    This is where reuse must fire, on both serve paths, and through the mux
    slot swap (conversations interleaved A1 B1 A2 B2 A3 B3).
    off = COLI_KV_PREFIX=0, the engine's own A/B hatch.

  reply-inclusive (max_tok=24, arm mux_agent): the full agent pattern — the
    model's reply is fed back and becomes part of the resent history. On the
    synthetic container the reply's id-stream is NOT recoverable from its
    text (tokenizer: 256 byte pieces vs a 4096-wide vocab; ids >= 256 decode
    to no bytes), so the all-or-nothing prefix match diverges at the first
    undecodable id. A synthetic-tokenizer artifact, not an engine bug: on a
    real BPE model every sampled id decodes to recoverable text and the
    resent bytes re-encode to the same ids. Real-model confirmation belongs
    to the gated Phase 5.

Per turn we record: prompt tokens (ACCEPT), client-side wall time to DONE,
and the [PREFIX] decision from stderr.

Usage: exp_prefix_q36.py [arm ...]
Writes ~/colibri-research/harness/results_prefix_q36.json.
"""
import subprocess, os, re, json, sys, time

ENGINE = os.path.expanduser(os.environ.get("PREFIX_ENGINE", "~/colibri/c/qwen36"))
SNAP   = os.path.expanduser(os.environ.get("PREFIX_SNAP", "~/synth_models/synth_qwen27"))
OUT    = os.path.expanduser(os.environ.get("PREFIX_OUT", "~/colibri-research/harness/results_prefix_q36.json"))
LOGDIR = os.path.expanduser(os.environ.get("PREFIX_LOGDIR", "~/colibri-research/harness"))
SCRUB = ("IOTH_", "SNAP", "CTX", "KVSAVE", "COLI_USAGE", "USAGE_SAVE", "PIN",
         "SERVE", "PROMPT", "Q36_", "KV8", "KV_TQ", "COLI_DENSE_BITS", "COLI_DENSE_INT4",
         "COLI_TIMERS", "N_NEW", "NGEN", "COLI_TEMP", "TRUNK_", "HOT_", "GRIFFIN",
         "KV_SLOTS", "COLI_KV_PREFIX", "COLI_PREFIX_LOG", "OPENAI")

T1 = ("You are a terminal agent. The user asks: list the files in the project "
      "directory and summarize the build system. Tool output follows. ") * 12   # ~1100 B
TOOL = ("\n[tool result %d] makefile found, 42 sources, gcc -O2 -fopenmp, "
        "target 'all' builds in 3s; tests green; disk 61%% used. Continue. ")

def base_for(conv):
    """Turn 1 of each conversation: the system prompt + task."""
    return (T1 + (f"\n[conversation {conv} system prompt tail] " if conv else "")).encode()

def run_arm(label, extra_env, max_tok=1, turns=3, convs=2):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(SCRUB):
            del env[k]
    env.update({"SNAP": SNAP, "Q36_MAXT": "4096", "OMP_NUM_THREADS": "4",
                "COLI_PREFIX_LOG": "1", "SERVE": "1"})
    env.update(extra_env)
    os.makedirs(LOGDIR, exist_ok=True)
    errlog = os.path.join(LOGDIR, f"prefix_{label}.err")
    p = subprocess.Popen([ENGINE, os.environ.get("PREFIX_CAP", "8"), os.environ.get("PREFIX_THREADS", "4")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=open(errlog, "wb"), env=env)
    # wait for the READY sentinel
    hdr = p.stdout.readline()
    while b"READY" not in hdr and hdr:
        hdr = p.stdout.readline()
    if not hdr:
        tail = open(errlog, errors="replace").read()[-800:] if os.path.exists(errlog) else ""
        print(f"[prefix] {label}: no READY sentinel; engine stderr tail:\n{tail}", file=sys.stderr)
        return {"error": "no READY sentinel", "label": label}
    rows = []
    # the agent pattern: each turn resends EVERYTHING the conversation has
    # produced so far (prompt + the model's own replies + tool results), then
    # appends the new tail. hist[conv] is exactly what a stateless client sends.
    hist = {c: base_for(c) for c in range(convs)}
    for turn in range(1, turns + 1):
        for conv in range(convs):
            rid = f"c{conv}t{turn}"
            payload = hist[conv]
            t0 = time.time()
            p.stdin.write(f"SUBMIT {rid} {conv} {len(payload)} {max_tok} 0.0 1.0\n".encode())
            p.stdin.write(payload + b"\n")
            p.stdin.flush()
            np = None
            gen = b""
            while True:
                line = p.stdout.readline()
                if not line:
                    return {"error": f"eof mid-turn {rid}", "rows": rows, "label": label}
                if line.startswith(b"ACCEPT "):
                    np = int(line.split()[2])
                elif line.startswith(b"DATA "):
                    n = int(line.split()[2])
                    gen += p.stdout.read(n); p.stdout.readline()   # payload + newline
                elif line.startswith(b"DONE "):
                    dt = round(time.time() - t0, 3)
                    stat = line.decode(errors="replace").strip()
                    rows.append({"id": rid, "conv": conv, "turn": turn, "np": np,
                                 "dt_s": dt, "stat": stat})
                    hist[conv] = hist[conv] + gen + (TOOL % turn).encode()
                    break
                elif line.startswith(b"ERROR "):
                    return {"error": line.decode(errors="replace").strip(), "rows": rows, "label": label}
    p.stdin.close()
    p.wait(timeout=60)
    # scrape the [PREFIX] decisions from stderr in order
    txt = open(errlog, errors="replace").read()
    dec = re.findall(r"\[PREFIX\] (reusing \d+ of \d+ prompt tokens \(\d+%\)|no reuse: .*)", txt)
    for r, d in zip(rows, dec):
        r["prefix"] = d
    results = {}
    if os.path.exists(OUT):
        try: results = json.load(open(OUT))
        except Exception: results = {}
    results[label] = {"rows": rows, "rc": p.returncode, "max_tok": max_tok}
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"== {label} (rc={p.returncode}, max_tok={max_tok}) ==", flush=True)
    for r in rows:
        print(f"  {r['id']}: np={r['np']:5d} dt={r['dt_s']:7.3f}s  {r.get('prefix','?')}", flush=True)
    return results[label]

def main():
    a = sys.argv[1:]
    arms = [
        # (label, env, max_tok) — max_tok 1: append-only; 24: reply-inclusive
        ("mux_on",     {"KV_SLOTS": "2"},                        1),
        ("mux_off",    {"KV_SLOTS": "2", "COLI_KV_PREFIX": "0"}, 1),
        ("single_on",  {},                                        1),
        ("single_off", {"COLI_KV_PREFIX": "0"},                   1),
        ("mux_agent",  {"KV_SLOTS": "2"},                       24),
    ]
    bad = 0
    for label, env, mt in arms:
        if a and label not in a:
            continue
        res = run_arm(label, env, max_tok=mt)
        if isinstance(res, dict) and res.get("error"):
            bad = 1
    sys.exit(bad)

if __name__ == "__main__":
    main()
