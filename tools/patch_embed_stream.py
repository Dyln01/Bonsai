#!/usr/bin/env python3
"""
patch_embed_stream.py  —  Q36_EMBED_STREAM=1

Replaces the full-resident f32 embedding load with a per-token row fetch.
Frees ~2.5–5 GB RAM (geometry dependent) with zero quality change.

Usage (from colibri/c after Bonsai overlay):
    python3 /path/to/bonsai/tools/patch_embed_stream.py
    # or:  python3 patch_embed_stream.py --dry-run
"""

import argparse, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # bonsai/
ENGINE = Path.home() / "colibri" / "c" / "qwen36.c"  # override with --engine

# ---------------------------------------------------------------------------
# Anchors (stable strings taken from the current engine surface)
# ---------------------------------------------------------------------------

# 1. Add the new fields to the model struct (near other embed / map fields)
ANCHOR_STRUCT = "float *embed;"
STRUCT_INSERT = """float *embed;
    int     embed_streamed;     /* Q36_EMBED_STREAM: 1 = row-fetch, 0 = resident */
    void   *embed_st;           /* safetensors handle when streamed */
    float  *embed_row;          /* single-row scratch (hidden,) */
"""

# 2. Replace the full-table load with a conditional
#    (the exact call site is the load_t_n / st_read_f32 path for the embedding)
ANCHOR_LOAD = "load_t_n(m, \"model.embed_tokens.weight\""
LOAD_REPLACEMENT = r'''/* ---- embedding load (patched by patch_embed_stream.py) ---- */
    m->embed_streamed = getenv("Q36_EMBED_STREAM") && getenv("Q36_EMBED_STREAM")[0] != '0';
    if (m->embed_streamed) {
        m->embed = NULL;
        m->embed_st = st_find(m->S, "model.embed_tokens.weight");
        if (!m->embed_st) {
            fprintf(stderr, "[qwen36] Q36_EMBED_STREAM: embed tensor not found\n");
            exit(1);
        }
        m->embed_row = (float*)aligned_alloc(64, (size_t)m->hidden * sizeof(float));
        if (!m->embed_row) { perror("embed_row"); exit(1); }
        fprintf(stderr, "[qwen36] embed STREAMED (row-fetch, frees resident table)\n");
    } else {
        load_t_n(m, "model.embed_tokens.weight\""""

# 3. Lookup site – replace direct pointer arithmetic with a row read
ANCHOR_LOOKUP = "x = m->embed + (size_t)token * m->hidden;"
LOOKUP_REPLACEMENT = """if (m->embed_streamed) {
        st_read_f32_row(m->embed_st, (size_t)token, m->embed_row, m->hidden);
        x = m->embed_row;
    } else {
        x = m->embed + (size_t)token * m->hidden;
    }"""

# 4. Free path
ANCHOR_FREE = "free(m->embed);"
FREE_REPLACEMENT = """if (m->embed_streamed) {
        free(m->embed_row);
        m->embed_row = NULL;
        /* embed_st is owned by the st context, do not free here */
    } else {
        free(m->embed);
    }
    m->embed = NULL;"""

def apply(path: Path, dry: bool = False) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    original = text

    def sub(anchor: str, repl: str, label: str) -> None:
        nonlocal text
        if anchor not in text:
            print(f"[warn] anchor not found for {label}:\n  {anchor[:80]}...")
            return
        text = text.replace(anchor, repl, 1)
        print(f"[ok]   {label}")

    sub(ANCHOR_STRUCT, STRUCT_INSERT, "struct fields")
    sub(ANCHOR_LOAD, LOAD_REPLACEMENT, "load path")
    sub(ANCHOR_LOOKUP, LOOKUP_REPLACEMENT, "token lookup")
    sub(ANCHOR_FREE, FREE_REPLACEMENT, "free path")

    if text == original:
        print("No changes made (anchors missing or already patched).")
        return

    if dry:
        print("--- dry-run: would write", path)
        return

    bak = path.with_suffix(path.suffix + ".pre_embed_stream")
    bak.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"[done] patched {path}")
    print(f"       backup  {bak}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, default=ENGINE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.engine.is_file():
        print(f"Engine not found: {args.engine}", file=sys.stderr)
        sys.exit(1)

    apply(args.engine, dry=args.dry_run)

if __name__ == "__main__":
    main()
