#!/usr/bin/env python3
"""fetch_wikitest2_raw.py — get wikitext-2-raw-v1 test split as a raw text file.

    fetch_wikitest2_raw.py [out_path]        (default: ./wiki.test.raw)

The historical S3 URL (s3.amazonaws.com/research.metamind.io/…) is dead — it
black-holes connections, which is why `wget` appeared to do nothing. The
datasets-server REST API is gone too. Fallback chain, first that works wins:

  1. `datasets` library (pip install datasets) — canonical, maintained;
  2. `pyarrow` + the dataset's converted-parquet branch (single 732 KB file,
     urllib only for the download);
  3. loud failure with the exact pip command, no silent half-files.

Reconstructs the raw file as "".join(rows["text"]) — each row keeps its
trailing newline, so the byte stream matches the original line structure.
Downloads ~1-25 MB depending on route; nothing else touches the network.
"""
import os
import sys
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "wiki.test.raw"
PARQUET = ("https://huggingface.co/datasets/salesforce/wikitext/resolve/"
           "refs%2Fconvert%2Fparquet/wikitext-2-raw-v1/test/0000.parquet")


def via_datasets():
    from datasets import load_dataset
    ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "".join(ds["text"])


def via_pyarrow():
    import io
    import pyarrow.parquet as pq
    req = urllib.request.Request(PARQUET, headers={"User-Agent": "bonsai-phase5"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    tbl = pq.read_table(io.BytesIO(raw))
    return "".join(tbl.column("text").to_pylist())


def main():
    text = None
    for name, fn in (("datasets", via_datasets), ("pyarrow+parquet", via_pyarrow)):
        try:
            text = fn()
            print("fetched via", name)
            break
        except ImportError:
            continue
        except Exception as e:
            print("route %s failed: %s" % (name, e), file=sys.stderr)
    if text is None:
        sys.exit("need one of: pip install datasets   OR   pip install pyarrow\n"
                 "(the old metamind S3 URL is dead; datasets-server REST is gone)")
    if len(text) < 100_000 or "\n" not in text:
        sys.exit("fetched text looks wrong (%d bytes) — refusing to write" % len(text))
    with open(OUT, "w", encoding="utf8") as f:
        f.write(text)
    lines = text.count("\n")
    print("wrote %s: %d bytes, %d lines" % (OUT, len(text), lines))


if __name__ == "__main__":
    main()
