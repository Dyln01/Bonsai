#!/usr/bin/env python3
"""check_runner_env.py — E11 mechanical oracle: runner env names vs engine getenv.

Bug class (hit three times in the field): a runner invokes the engine BINARY
directly but sets a LAUNCHER alias (Q36_HOT_FRAC / Q36_GRIFFIN — names
q36-minipc.sh maps to HOT_FRAC / GRIFFIN). The binary never reads the alias,
silently runs its stock default, and the arm is vacuous — or, when the missing
name is TRUNK_MAP=1, the stock loader pulls the whole dense trunk into RAM and
an 8 GB box OOM-kills it mid-load (battery_llm-030_20261023: seven arms
rc=137 at an identical ~5.1 GB fs-read watermark).

Modes:
  loose (default, CI gates): scans the direct-binary runners
    (tools/run_battery.sh, tools/run_phase5.sh). Known names = getenv set of
    engine/qwen36.c + ALLOW (upstream/libgomp/launcher-facing, each verified
    by hand against pinned colibri bf24429 and re-verified in strict mode).
  strict (--colibri DIR, CI engine-build): additionally unions the getenv set
    of the pinned colibri checkout, RE-VERIFIES the ALLOW upstream claims
    against it, and also scans tools/q36-minipc.sh (its MoE COMMON uses
    colibri-core names like CAP_RAISE/PIN/USAGE_SAVE that only strict mode
    can vouch for).

Rules:
  1. Every env NAME=value token on an engine-invocation line (COMMON=...,
     arm/run_arm calls, `env ...` lines incl. backslash continuations) must be
     in the known set. BATTERY_*/PHASE5_* runner knobs and ${...} expansions
     are excluded by construction.
  2. No ALLOW name may be getenv'd by qwen36.c itself (stale allowlist).
  3. Each runner's full COMMON definition (the one carrying SNAP=) must
     contain TRUNK_MAP=1 — dense trunks stay mmap'd on disk.
  4. Known decoys (Q36_HOT_FRAC=, Q36_GRIFFIN=) get an actionable message.

Exit 0 = pass; 1 = any violation (printed per-file with line numbers).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine" / "qwen36.c"
RUNNERS = [ROOT / "tools" / "run_battery.sh", ROOT / "tools" / "run_phase5.sh"]
LAUNCHER = ROOT / "tools" / "q36-minipc.sh"  # strict mode only

# Names on engine lines that qwen36.c does not getenv() itself. Upstream
# claims verified against pinned colibri bf24429 (c/ getenv union); strict
# mode re-verifies ALLOW_UPSTREAM every run.
ALLOW_UPSTREAM = {
    "COLI_TEMP": "colibri core — sampling temperature",
    "COLI_NO_OMP_TUNE": "colibri core — OMP thread-probe suppression",
    "KVSAVE": "colibri core — KV slot persistence",
}
ALLOW = dict(ALLOW_UPSTREAM)
ALLOW["OMP_NUM_THREADS"] = "libgomp (runtime, not engine source)"
ALLOW["Q36_CORPUS"] = "launcher-facing (q36-minipc calibrate reads it itself)"

DECOYS = {"Q36_HOT_FRAC": "HOT_FRAC", "Q36_GRIFFIN": "GRIFFIN"}
CARRIER_VARS = {"COMMON"}  # shell carriers, not engine env

# NAME= not preceded by $ { . : or alnum (skips ${VAR:-x}, $NAME, path bits).
ASSIGN = re.compile(r"(?<![A-Za-z0-9_$.{:])([A-Z][A-Z0-9_]{2,})=")
# Lines that reach an engine process. `env` as a word starts launcher-style
# invocations; COMMON= catches definitions and appends; arm/run_arm catches
# battery/phase5 arm calls (even mid-line after `if ...; then`).
INVOCATION = re.compile(r"\bCOMMON=|\b(run_)?arm[\s(]|\benv\b")
KNOB_PREFIXES = ("BATTERY_", "PHASE5_")


def getenv_names(tree: Path, patterns=("*",)) -> set:
    names = set()
    for pat in patterns:
        for f in tree.rglob(pat):
            if f.suffix in (".c", ".h") and f.is_file():
                names.update(re.findall(r'getenv\("([A-Z0-9_]+)"\)',
                                        f.read_text(errors="replace")))
    return names


def scanned_lines(text: str, track_env: bool):
    """Yield (lineno, line) for engine-invocation lines, following backslash
    continuations. track_env=False (loose) ignores bare `env` starts — the
    direct runners only invoke via `env $COMMON`/arm helpers, and their echo
    strings mention 'env' freely."""
    continuing = False
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        hit = bool(s) and not s.startswith("#") and (
            continuing or INVOCATION.search(line)
            and (track_env or re.search(r"\benv \$COMMON\b|\bCOMMON=|\b(run_)?arm[\s(]", line)))
        if hit:
            yield i, line
        continuing = hit and s.endswith("\\")


def check_file(path: Path, known: set, track_env: bool, fails: list,
               require_trunk_map: bool = True) -> None:
    text = path.read_text(errors="replace")
    rel = path.relative_to(ROOT)
    seen_common = False
    for lineno, line in scanned_lines(text, track_env):
        if "COMMON=" in line and "SNAP=" in line:
            seen_common = True
            # rule 3: dense-dialect COMMON must carry TRUNK_MAP=1. The launcher
            # composes TWO COMMONs (MoE at :70 has no dense markers — exempt);
            # a dense line is recognized by HOT_FRAC/GRIFFIN in it.
            dense = require_trunk_map or "HOT_FRAC" in line or "GRIFFIN" in line
            if dense and "TRUNK_MAP=1" not in line:
                fails.append(f"{rel}:{lineno}: COMMON lacks TRUNK_MAP=1 (dense trunk "
                             f"must stay mmap'd; the stock loader OOMs small-RAM boxes)")
        for name in ASSIGN.findall(line):
            if name in CARRIER_VARS or name.startswith(KNOB_PREFIXES):
                continue
            if name in DECOYS:
                fails.append(f"{rel}:{lineno}: DECOY {name}= — the binary reads "
                             f"{DECOYS[name]}; the alias is launcher-only (E11)")
            elif name not in known:
                fails.append(f"{rel}:{lineno}: {name}= is not getenv'd by engine/qwen36.c"
                             + (" / pinned colibri" if track_env else "")
                             + f" and not in ALLOW — engine will ignore it (vacuous arm?)")
    if not seen_common:
        fails.append(f"{rel}: no COMMON= definition (with SNAP=) found — scanner drifted")


def main(argv) -> int:
    colibri = None
    if "--colibri" in argv:
        colibri = Path(argv[argv.index("--colibri") + 1])
        if not colibri.is_dir():
            print(f"FAIL: --colibri {colibri} is not a directory")
            return 1

    eng = getenv_names(ENGINE.parent, ("qwen36.c",))
    if not eng:
        print(f"FAIL: no getenv names found in {ENGINE} — oracle broken")
        return 1
    fails = []

    # rule 2: stale allowlist entries
    for name in sorted(set(ALLOW) & eng):
        fails.append(f"allowlist stale: {name} IS getenv'd by qwen36.c — drop it from ALLOW")

    known = eng | set(ALLOW)
    if colibri:
        up = getenv_names(colibri / "c")
        if not up:
            print(f"FAIL: no getenv names under {colibri}/c — wrong checkout?")
            return 1
        for name in sorted(set(ALLOW_UPSTREAM) - up):
            fails.append(f"ALLOW claims upstream but pinned colibri does not read: {name}")
        known |= up
        check_file(LAUNCHER, known, True, fails, require_trunk_map=False)
    for runner in RUNNERS:
        check_file(runner, known, bool(colibri), fails)

    if fails:
        print(f"check_runner_env: FAIL ({len(fails)})")
        for f in fails:
            print("  " + f)
        return 1
    mode = "strict" if colibri else "loose"
    n_files = len(RUNNERS) + (1 if colibri else 0)
    print(f"check_runner_env: PASS [{mode}] — {n_files} shell files, "
          f"{len(eng)} engine env names, {len(known)} known, {len(ALLOW)} allowlist")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
