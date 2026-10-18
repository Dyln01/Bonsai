#!/usr/bin/env python3
"""patch_mtp_oracle.py — Phase 2b part 3a: forced-accept oracle gates for MTP.

  - Q36_MTP_FORCE=accept|mixed|cycle|rowN now drives the MTP proposal path:
    mapped onto sp->force at spec_begin when the head is loaded (the oracle is
    ref.json's full_ids, exactly like COLI_LOOKUP_FORCE), so q36_spec_forced
    feeds the head's proposal slot with oracle/wrong tokens per mode
  - DUMP_IDS=<path> test hook: the generated id sequence (prompt+continuation),
    one per line — CI bootstraps ref.json from a stock run with it
  - acceptance counting moves to q36_spec_settle (keep-1 per verify, the same
    number the gate books), replacing the per-match increment, so the
    `[qwen36 MTP]` acceptance line equals the gate's accepted/drafts

Gates (CI): FORCE=accept => acceptance 100.0% AND last-logit sha == stock
(correctness proof of verify + head-row rollback with everything accepted);
FORCE=mixed/cycle => logits still == stock (rejections + partial accepts).

Anchored exact-byte edits, house style. Backup qwen36.c.preoracle.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "DUMP_IDS" in src:
    raise SystemExit("patch_mtp_oracle: already applied")
if "q36_mtp_propose" not in src:
    raise SystemExit("patch_mtp_oracle: part 2 (patch_mtp_draft.py) must be applied first")

EDITS = []
def edit(old, new):
    EDITS.append((old, new))

# 1. Q36_MTP_FORCE -> sp->force when the head is loaded
edit("""    sp->force = q36_spec_force_mode(getenv("COLI_LOOKUP_FORCE"), &sp->force_row);
""",
     """    sp->force = q36_spec_force_mode(getenv("COLI_LOOKUP_FORCE"), &sp->force_row);
    if (!sp->force && g_q36_mtp_force[0] && m->mtp && m->mtp->loaded)
        sp->force = q36_spec_force_mode(g_q36_mtp_force, &sp->force_row);   /* Q36_MTP_FORCE */
""")

# 2. acceptance counting at settle (same number the gate books), not per match
edit("""            sp->accepted++;
            if (sp->cur_src == SPEC_SRC_MTP) sp->m_accepted++;
""",
     """            sp->accepted++;
""")
edit("""    q36_mtp_settle(m, sp->ahead_pos, keep);
""",
     """    q36_mtp_settle(m, sp->ahead_pos, keep);
    if (sp->cur_src == SPEC_SRC_MTP) sp->m_accepted += (uint64_t)(keep > 0 ? keep - 1 : 0);
""")

# 3. DUMP_IDS test hook (CI bootstraps ref.json from a stock run)
edit("""    q36_spec_end(m, sp);
}
""",
     """    q36_spec_end(m, sp);
    { const char *ip = getenv("DUMP_IDS");
      if (ip && *ip) { FILE *f = fopen(ip, "w");
        if (f) { for (int i = 0; i < len; i++) fprintf(f, "%d\\n", out[i]);
                 fclose(f); fprintf(stderr, "[dump] wrote %d ids -> %s\\n", len, ip); }
        else fprintf(stderr, "[dump] cannot open %s\\n", ip); } }
}
""")

for old, new in EDITS:
    if src.count(old) != 1:
        raise SystemExit("patch_mtp_oracle: anchor count=%d (want 1): %r" % (src.count(old), old[:70]))
    src = src.replace(old, new, 1)

bak = P + ".preoracle"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_mtp_oracle applied (%d edits, backup %s)" % (len(EDITS), bak))
