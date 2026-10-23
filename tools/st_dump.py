#!/usr/bin/env python3
"""st_dump.py — inspect a colibri container's safetensors shards the way c/st.h sees them.

Read-only probe (headers only; tensor bytes are never touched). For every
*.safetensors in the directory it reports, per entry: dtype, shape, numel,
data_offsets span, and the checks the engine applies at load:

  * span == numel * esz for every dtype EXCEPT U8/I8 (st_init's cross-check;
    raw quant bytes are read by byte count, so their shape may legitimately
    disagree with their span),
  * data_offsets inside the file,
  * a dtype st_dtype_code accepts at all,
  * tensor names defined in more than one shard — which st_init REFUSES unless
    model.safetensors.index.json maps the name to exactly one of the copies
    ("the copy the index maps is read, the other ignored").

Why this tool exists: the engine and the Python container tools trust different
header fields (load_t_n checks the SHAPE-derived numel; hot_sort.py reads the
byte SPAN), and a directory can hold more files than the engine ended up
reading. When those disagree the engine runs fine while a tool chokes — this
dump prints both views side by side so the disagreement is a paste-able fact
instead of a guessing game.

Usage:
  python3 tools/st_dump.py <container_dir> [name_filter] [--all]
    name_filter  substring; matching entries print in full (default: only
                 summary lines + every entry that trips a check)
    --all        print every entry of every shard

Exit status: 0 = no issues found, 1 = at least one issue (useful in scripts).
Stdlib only.
"""
import json
import os
import struct
import sys
import time

# st.h st_dtype_code / st_dtype_esz, mirrored. Names must stay in sync with the
# pinned engine (c/st.h); an unknown dtype string here is itself an issue
# (st_dtype_code would coli_load_fail on it).
ESZ = {
    "BF16": 2, "F16": 2, "F32": 4,
    "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E4M3FN": 1, "float8_e4m3fn": 1,
    "F8_E8M0": 1, "F8_E8M0FNU": 1,
    "I64": 8, "U64": 8, "I32": 4, "U32": 4,
}
# st_init's span cross-check exempts exactly dtype code 3 (U8/I8).
SPAN_EXEMPT = ("U8", "I8")


def read_header(path):
    """Return (header_len, entries, intra_dups). entries preserves file order;
    a name appearing twice inside ONE header JSON is reported in intra_dups
    (Python dicts silently keep the last copy — st.h's parser sees both)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(hlen)
    hdr = json.loads(raw)
    keys = [k for k, _ in json.loads(raw, object_pairs_hook=lambda p: p)]
    dups = sorted({k for k in keys if keys.count(k) > 1 and k != "__metadata__"})
    entries = [(k, hdr[k]) for k in dict.fromkeys(keys) if k != "__metadata__"]
    return hlen, entries, dups, ("__metadata__" in hdr)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    all_entries = "--all" in sys.argv
    if not args:
        print(__doc__)
        return 2
    d = args[0]
    filt = args[1] if len(args) > 1 else None
    if not os.path.isdir(d):
        print(f"st_dump: not a directory: {d}", file=sys.stderr)
        return 2

    issues = []

    def issue(msg):
        issues.append(msg)
        print(f"  !! {msg}")

    print(f"== directory: {d}")
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            st = os.stat(p)
            print(f"  {st.st_size:>14,} B  {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}  {name}")
        else:
            print(f"  {'':>14}   {'':>16}  {name}/")

    shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    idx_path = os.path.join(d, "model.safetensors.index.json")
    weight_map = {}
    if os.path.exists(idx_path):
        try:
            weight_map = json.load(open(idx_path)).get("weight_map") or {}
            print(f"== index: model.safetensors.index.json ({len(weight_map)} name(s) mapped)")
        except Exception as e:  # a broken index is itself the finding
            issue(f"model.safetensors.index.json exists but does not parse: {e}")
    else:
        print("== index: no model.safetensors.index.json")

    owner = {}   # tensor name -> list of shards defining it
    for f in shards:
        path = os.path.join(d, f)
        fsz = os.path.getsize(path)
        try:
            hlen, entries, intra_dups, has_meta = read_header(path)
        except Exception as e:
            issue(f"{f}: header does not parse ({e})")
            continue
        data_start = 8 + hlen
        n_bad = 0
        print(f"== {f}: {fsz:,} B, header {hlen:,} B, {len(entries)} tensor(s)"
              + (", __metadata__ present" if has_meta else ""))
        for k in intra_dups:
            issue(f"{f}: tensor '{k}' appears TWICE inside the same header JSON "
                  f"(last copy wins for Python readers; st.h indexes both)")
        for name, e in entries:
            owner.setdefault(name, []).append(f)
            dt = e.get("dtype")
            shape = e.get("shape")
            offs = e.get("data_offsets")
            bad = []
            span = "?"
            if not isinstance(dt, str) or not isinstance(shape, list) or \
                    not (isinstance(offs, list) and len(offs) == 2):
                bad.append(f"malformed entry {e!r}")
            else:
                # data_offsets are [begin, END) within the data region
                begin, end = offs[0], offs[1]
                span = end - begin
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                if dt not in ESZ:
                    bad.append(f"unsupported dtype '{dt}' (st_dtype_code refuses)")
                else:
                    esz = ESZ[dt]
                    if dt not in SPAN_EXEMPT and span != numel * esz:
                        bad.append(f"span/shape disagree: {numel} elements x {esz} B = "
                                   f"{numel * esz:,} B but data_offsets span {span:,} B "
                                   f"(st_init refuses float dtypes like this)")
                    if not (isinstance(begin, int) and isinstance(end, int)) or \
                            begin < 0 or end < begin or data_start + end > fsz:
                        bad.append(f"data_offsets [{begin},{end}) out of file bounds ({fsz:,} B)")
            if all_entries or bad or (filt is not None and filt in name):
                print(f"  {name}  {dt}  shape={shape}  span={span:,} B" if not bad
                      else f"  {name}  {e!r}")
            for b in bad:
                n_bad += 1
                issue(f"{f}: tensor '{name}': {b}")
        if filt is not None and not all_entries:
            hit = sum(1 for name, _ in entries if filt in name)
            print(f"  ({hit} entry/entries match '{filt}')" if hit else
                  f"  (no entry matches '{filt}')")

    # cross-shard duplicates, resolved exactly the way st_init resolves them
    dups = {n: fs for n, fs in owner.items() if len(fs) > 1}
    if dups:
        print(f"== duplicate tensor names across shards: {len(dups)}")
        for n, fs in sorted(dups.items()):
            auth = weight_map.get(n)
            if auth in fs:
                print(f"  {n}: in {fs} — index maps it to '{auth}': engine reads that copy, "
                      f"ignores the other(s)")
            elif auth:
                issue(f"tensor '{n}' is in {fs} but the index maps it to '{auth}' "
                      f"(a third file) — index and files disagree; st_init refuses")
            else:
                why = ("the index does not list it" if weight_map else
                       "there is no model.safetensors.index.json")
                issue(f"tensor '{n}' is defined in {len(fs)} shards ({', '.join(fs)}) and "
                      f"{why} — st_init REFUSES to load this directory; Python tools that "
                      f"glob *.safetensors see one of the copies")
    else:
        print("== duplicate tensor names across shards: none")

    total = sum(len(fs) for fs in owner.values())
    print(f"== {len(shards)} shard(s), {len(owner)} distinct tensor name(s), {total} entries")
    print(f"== issues: {len(issues)}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
