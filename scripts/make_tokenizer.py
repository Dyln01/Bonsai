#!/usr/bin/env python3
"""make_tokenizer.py — minimal tokenizer.json for the synthetic containers.

The engine's tok.h reads model.vocab (string->id), optional model.merges, and
added_tokens. A byte-level tokenizer with 256 single-codepoint entries (ids
0..255, where entry i is chr(i) for printable ASCII and the byte-level
representation otherwise) plus an added EOS token makes PROMPT-mode runs
work: any text encodes to ids < 256 < vocab_size (2048/4096).

Note: chr(0) can't be embedded in the JSON string, so bytes 0 is mapped to a
placeholder; PROMPT text never contains NUL anyway.
"""
import json, os, sys

def make(path, eos_id):
    vocab = {}
    # byte-level: bytes 33..126 printable ASCII map to themselves; for the rest,
    # use the standard byte-level unicode mapping (U+0100 + byte) as 2-char UTF-8.
    for b in range(256):
        if 33 <= b <= 126:
            vocab[chr(b)] = b
        else:
            vocab[chr(0x100 + b)] = 256 + b if b == 0 else (256 + b if b < 33 else 256 + b)
    # Simpler: bytes >= 127 and < 33 -> chr(0x100 + b) id 256+b; byte 0 unused.
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": eos_id, "content": "<|end_of_text|>", "special": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": False},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": None,
            "vocab": vocab,
            "merges": [],
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"wrote {path}: 256 byte tokens + EOS id {eos_id}")

if __name__ == "__main__":
    eos = int(sys.argv[2]) if len(sys.argv) > 2 else 2047
    make(sys.argv[1], eos)
