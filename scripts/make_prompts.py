#!/usr/bin/env python3
"""make_prompts.py — regenerate the battery's historical prompt fixtures, byte-exact.

    make_prompts.py [out_dir]      (default: ~/eval)

Writes:
  prompt_8k.txt        "alpha beta ... kappa " * 280    (~8k tokens; the KV8/KVTQ/
                                                          GQA-absorb battery prompt)
  prompt_long.txt      same sentence * 1200             (~68.4k tokens; the long-ctx
                                                          fixture behind the 57.65 ms/tok
                                                          scan number and the 3a battery)
  prompt_distinct_N.txt "w0 w1 ... w{N-1}" for N in 512, 2048, 8192
                                                          (distinct byte-tokens; the
                                                          decode-scaling lookup arms)

These are the EXACT strings the original ad-hoc experiment scripts used
(exp_qwen27_kv8b.py, exp_qwen27_kv8.py, diag_decode_scaling.py), so new battery
numbers stay comparable with docs/KV8_QWEN36.md, docs/KVTQ_QWEN36.md,
docs/GQA_ABSORB_QWEN36.md and docs/SCAN_DECAY_QWEN36.md. Deterministic, no rng
state, no downloads.
"""
import os
import sys

SENT = "alpha beta gamma delta epsilon zeta eta theta iota kappa "


def main():
    out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/eval")
    os.makedirs(out, exist_ok=True)
    files = {"prompt_8k.txt": SENT * 280,
             "prompt_long.txt": SENT * 1200}
    for n in (512, 2048, 8192):
        files["prompt_distinct_%d.txt" % n] = " ".join("w%d" % i for i in range(n))
    for name, txt in files.items():
        open(os.path.join(out, name), "w").write(txt)
        print("%-24s %7d bytes" % (name, len(txt)))
    print("battery prompt: %s/prompt_8k.txt" % out)


if __name__ == "__main__":
    main()
