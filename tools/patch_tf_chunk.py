#!/usr/bin/env python3
"""patch_tf_chunk.py — chunked teacher-forced scoring for PPL=1 (E16).

tf_nll scored every token with a single-token step(): on a field host where
each decode step re-streams the cold trunk (llm-030: 2225 s/token measured,
ROADMAP E13) that is ~52 DAYS per arm at 2048 scored tokens — Phase 5 was
arithmetically unrunnable as scripted. Q36_TF_CHUNK=S (S>1) scores S
positions per weight pass through step_ex's nlogits rows — the exact
multi-row kernel speculative verify runs, whose decode-exactness the CI MTP
forced-accept gates already prove byte-for-byte. The per-scored-token floor
becomes one lm_head pass (~90 s field) instead of a full trunk+head stream
(~37 min). Default (unset or 1) keeps the per-token path bit-identical.

Anchored exact-byte edit, house style. Backup qwen36.c.pretfc.
"""
import os
import shutil
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/colibri/c/qwen36.c")
src = open(P, "rb").read().decode("utf8", "surrogateescape")
if "q36_tf_chunk" in src:
    raise SystemExit("patch_tf_chunk: already applied")

OLD = r"""static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    if (nfull > QWEN36_ATTN_MAX_CTX) {
        fprintf(stderr, "[ctx] %d tokens exceed the %d-token ceiling\n",
                nfull, QWEN36_ATTN_MAX_CTX);
        exit(1);
    }
    m->max_t = nfull;
    reset_recurrent(m);
    ensure_kv(m);
    m->kv_len = 0;
    double nll = 0; int scored = 0;
    float *logit = step(m, full, np, 0);
    for (int i = np; i < nfull; i++) {
        float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[full[i]] - mx - log(Z));
        scored++;
        free(logit); logit = NULL;
        if (i == nfull - 1) break;
        logit = step(m, &full[i], 1, i);
    }
    if (logit) free(logit);
    *nll_out = nll / scored;
    return scored;
}
"""

NEW = r"""/* E16: chunked teacher forcing (Q36_TF_CHUNK=S, S>1). Scoring position i
 * needs the logits row at position i-1; per-token stepping re-streams the
 * whole cold trunk for EVERY scored token (llm-030: 2225 s/token, E13 ->
 * 52 days per arm at 2048 scored). Feeding S tokens through step_ex with
 * nlogits=S scores S positions from ONE weight pass, on the same multi-row
 * kernel speculative verify runs (the MTP forced-accept gates prove it
 * decode-exact). Default 1 = the per-token path, unchanged. */
static int q36_tf_chunk(void) {
    static int g = -1;
    if (g < 0) { const char *e = getenv("Q36_TF_CHUNK"); g = (e && *e) ? atoi(e) : 1; if (g < 1) g = 1; }
    return g;
}
static double q36_tf_row_nll(const float *logit, const Cfg *c, int token) {
    float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
    double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
    return -((double)logit[token] - mx - log(Z));
}
static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    if (nfull > QWEN36_ATTN_MAX_CTX) {
        fprintf(stderr, "[ctx] %d tokens exceed the %d-token ceiling\n",
                nfull, QWEN36_ATTN_MAX_CTX);
        exit(1);
    }
    m->max_t = nfull;
    reset_recurrent(m);
    ensure_kv(m);
    m->kv_len = 0;
    double nll = 0; int scored = 0;
    const int chunk = q36_tf_chunk();
    int passes = 1;
    float *logit = step(m, full, np, 0);   /* the row at position np-1 scores full[np] */
    for (int i = np; i < nfull; ) {
        nll += q36_tf_row_nll(logit, c, full[i]); scored++;   /* held row = position i-1 */
        if (i == nfull - 1) { free(logit); logit = NULL; break; }
        if (chunk <= 1) {
            free(logit);
            logit = step(m, &full[i], 1, i);
            i++;
            continue;
        }
        int S = nfull - i; if (S > chunk) S = chunk;
        free(logit); logit = NULL;
        float *rows = step_ex(m, &full[i], S, i, S);   /* rows at positions i..i+S-1 */
        passes++;
        for (int r = 0; r + 1 < S; r++)   /* row r scores full[i+r+1] */
            nll += q36_tf_row_nll(rows + (int64_t)r * c->vocab, c, full[i + r + 1]), scored++;
        logit = malloc((size_t)c->vocab * sizeof(float));   /* carry the last row */
        memcpy(logit, rows + (int64_t)(S - 1) * c->vocab, (size_t)c->vocab * sizeof(float));
        free(rows);
        i += S;
    }
    if (logit) free(logit);
    *nll_out = scored ? nll / scored : 0.0;
    if (chunk > 1)
        fprintf(stderr, "[tf] chunk=%d: %d scored tokens in %d weight passes (%d per-token)\n",
                chunk, scored, passes, scored);
    return scored;
}
"""

if src.count(OLD) != 1:
    raise SystemExit("patch_tf_chunk: anchor count=%d (want 1)" % src.count(OLD))
src = src.replace(OLD, NEW, 1)

bak = P + ".pretfc"
shutil.copy(P, bak)
open(P, "wb").write(src.encode("utf8", "surrogateescape"))
print("patch_tf_chunk applied (1 edit, backup %s)" % bak)
