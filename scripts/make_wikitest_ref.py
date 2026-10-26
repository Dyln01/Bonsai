#!/usr/bin/env python3
"""make_wikitest_ref.py — build the Phase 5 perplexity ref.json from wikitext-2 raw.

    make_wikitest_ref.py <container_dir> <wiki.test.raw> <out.json>
                         [--ctx 512] [--len 4096] [--bytes 60000]

Downloads NOTHING (Phase 5 rule): you place wiki.test.raw locally yourself
after sign-off. Tokenization is delegated to the ENGINE itself so the ids are
exactly what tf_nll will score: run prompt mode with N_NEW=1 and DUMP_IDS,
then drop the single generated id — the rest is the pure wikitext encoding.

Output ref.json = {"prompt_ids": first --ctx ids, "full_ids": first --len ids},
the contract engine ref-mode/PPL=1 expects (tf_nll scores positions
ctx..len-1 — so --len must be ctx + the wanted SCORED count; run_phase5.sh
passes --len=CTX+LEN). Prints the exact scoring command afterwards.

The engine run is a full prefill of the slice (DUMP_IDS lands at the end of
generate()); on a streaming host that costs a weight pass — size --bytes to
just what --ctx/--len need (~6 bytes/token). The ids themselves are
forward-pass independent: generate() copies the prompt ids verbatim before
any step, so HOT_FRAC/pruning cannot taint ref.json — but the call must still
FIT the box: E15, run it with TRUNK_MAP=1 + a fitting HOT_FRAC or the stock
loader mallocs the whole dense trunk and an 8 GB host kills it in seconds.

Env: Q36_ENGINE (default ~/colibri/c/qwen36), Q36_CAP (default 8),
Q36_THREADS (default 4).
"""
import json
import os
import subprocess
import sys
import tempfile


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=") + [True])[:2] if "=" in a else (a[2:], True)
                for a in sys.argv[1:] if a.startswith("--"))
    if len(args) != 3:
        sys.exit(__doc__)
    container, raw, out = args
    ctx = int(opts.get("ctx", 512))
    length = int(opts.get("len", 4096))
    nbytes = int(opts.get("bytes", 60000))
    engine = os.path.expanduser(os.environ.get("Q36_ENGINE", "~/colibri/c/qwen36"))
    cap = os.environ.get("Q36_CAP", "8")
    thr = os.environ.get("Q36_THREADS", "4")

    text = open(raw, encoding="utf8", errors="replace").read()
    # wikitext-2 raw has loose line structure; collapse whitespace runs but keep
    # sentence flow so byte-level and BPE tokenizers both behave normally
    text = " ".join(text.split())
    slice_ = text[:nbytes]
    if len(slice_) < 2000:
        sys.exit("wiki.test.raw slice too small (%d bytes)" % len(slice_))

    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "wiki_slice.txt")
        idf = os.path.join(td, "ids.txt")
        open(pf, "w", encoding="utf8").write(slice_)
        env = dict(os.environ)
        env.update({"SNAP": container, "N_NEW": "1", "COLI_TEMP": "0",
                    "DUMP_IDS": idf, "KVSAVE": "0", "COLI_NO_OMP_TUNE": "1",
                    "OMP_NUM_THREADS": thr, "NOSTREAM": "1"})
        r = subprocess.run([engine, cap, thr, pf], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("engine rc=%d\n%s\n"
                     "hint (E15): on small-RAM hosts this engine call needs the streaming "
                     "dialect — TRUNK_MAP=1 plus a HOT_FRAC that fits (tools/run_phase5.sh "
                     "sets both; without TRUNK_MAP the stock loader mallocs the whole "
                     "dense trunk and the box kills it)."
                     % (r.returncode, r.stderr[-800:]))
        if not os.path.exists(idf):
            sys.exit("no DUMP_IDS output — engine too old? need the DUMP_IDS hook")
        ids = [int(l) for l in open(idf) if l.strip()]
    if len(ids) < 2:
        sys.exit("engine returned %d ids" % len(ids))
    ids = ids[:-1]                      # drop the one generated token
    if len(ids) < length:
        sys.exit("only %d wikitext tokens encoded, need %d (ctx %d + %d scored); "
                 "raise --bytes (now %d)" % (len(ids), length, ctx, length - ctx, nbytes))
    full = ids[:length]
    ref = {"prompt_ids": full[:ctx], "full_ids": full}
    json.dump(ref, open(out, "w"))
    print("ref.json: %d prompt (ctx) + %d scored tokens -> %s"
          % (ctx, len(full) - ctx, out))
    print("score it with:")
    print('  SNAP=%s PPL=1 COLI_TEMP=0 %s %s %s %s'
          % (container, engine, cap, thr, out))


if __name__ == "__main__":
    main()
