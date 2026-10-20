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
ctx..len-1). Prints the exact scoring command afterwards.

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
            sys.exit("engine rc=%d\n%s" % (r.returncode, r.stderr[-800:]))
        if not os.path.exists(idf):
            sys.exit("no DUMP_IDS output — engine too old? need the DUMP_IDS hook")
        ids = [int(l) for l in open(idf) if l.strip()]
    if len(ids) < 2:
        sys.exit("engine returned %d ids" % len(ids))
    ids = ids[:-1]                      # drop the one generated token
    if len(ids) < ctx + 64:
        sys.exit("only %d wikitext tokens encoded; raise --bytes" % len(ids))
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
