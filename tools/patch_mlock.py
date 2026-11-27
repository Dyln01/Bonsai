#!/usr/bin/env python3
"""patch_mlock.py - Q36_MLOCK=1: pin the hot trunk with mlock() after the TRUNK_WARM madvise pass.

madvise(WILLNEED) is advisory: under memory pressure the kernel may drop the pages, and the field log
shows the same config at 52 s and 531 s/token when the page cache thrashes. Q36_MLOCK=1 locks the warmed
ranges (best-effort, capped by Q36_MLOCK_MB, default 70 % of physical RAM) so the resident set stays
resident. Off by default: no behaviour change unless the env var is set. Bit-identical output.

Two anchored edits + one include, house style. Backup qwen36.c.premlock.
"""
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "Q36_MLOCK" in src:
    raise SystemExit("patch_mlock: already applied")

OLD1 = """        double cover = q36_hot_frac();
        double w0 = now_s(); int64_t warmed = 0;"""
NEW1 = """        double cover = q36_hot_frac();
        double w0 = now_s(); int64_t warmed = 0;
        const char *mlk_e = getenv("Q36_MLOCK");
        int mlk_on = (mlk_e && *mlk_e == '1'), mlk_failed = 0;
        int64_t mlk_bytes = 0, mlk_cap = 0;
        if (mlk_on) {
            const char *mb = getenv("Q36_MLOCK_MB");
            long long phys = (long long)sysconf(_SC_PHYS_PAGES) * (long long)sysconf(_SC_PAGESIZE);
            mlk_cap = (mb && *mb) ? (int64_t)atoll(mb) * 1024 * 1024 : (int64_t)(phys * 0.70);
        }"""
OLD2 = """#ifdef MADV_WILLNEED
                madvise(base, span, MADV_WILLNEED);
#endif
                warmed += span;
            }
        }
        fprintf(stderr, "[warm] madvise(WILLNEED) over %.2f GB of mapped trunk (frac %.2f)\\n",
                warmed/1e9, cover);
    }"""
NEW2 = """#ifdef MADV_WILLNEED
                madvise(base, span, MADV_WILLNEED);
#endif
                /* Q36_MLOCK=1: pin the hot trunk so the kernel cannot evict it under pressure
                 * (WILLNEED is only advisory; the field log shows the same config at 52 s and
                 * 531 s/token when the cache thrashes). Best-effort, capped by Q36_MLOCK_MB
                 * (default 70% of physical RAM) so it can never lock the box into OOM. */
                if (mlk_on && mlk_bytes + span <= mlk_cap) {
                    if (mlock(base, (size_t)span) == 0) mlk_bytes += span;
                    else if (!mlk_failed) {
                        mlk_failed = 1;
                        fprintf(stderr, "[warm] mlock failed (%s) - raise RLIMIT_MEMLOCK "
                                "(ulimit -l unlimited / LimitMEMLOCK=infinity); continuing unpinned\\n",
                                strerror(errno));
                    }
                }
                warmed += span;
            }
        }
        fprintf(stderr, "[warm] madvise(WILLNEED) over %.2f GB of mapped trunk (frac %.2f)\\n",
                warmed/1e9, cover);
        if (mlk_on)
            fprintf(stderr, "[warm] mlock pinned %.2f GB of %.2f GB cap%s\\n", mlk_bytes/1e9, mlk_cap/1e9,
                    mlk_bytes < warmed ? " (rest left to the page cache)" : "");
    }"""
OLD3 = "#include <sys/mman.h>   /* TRUNK_MAP: mmap-backed dense weights */"
NEW3 = OLD3 + "\n#include <errno.h>      /* Q36_MLOCK error report */"
for o in (OLD1, OLD2, OLD3):
    if src.count(o) != 1:
        raise SystemExit("patch_mlock: anchor not found exactly once:\n" + o[:80])
open(P + ".premlock", "wb").write(src.encode("utf8", "surrogateescape"))
for o, n in ((OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)):
    src = src.replace(o, n, 1)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mlock: applied (backup %s.premlock)" % P)
