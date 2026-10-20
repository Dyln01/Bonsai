/* scan_decay.h — Phase 3b-scan: two-tier deterministic decode scan.
 *
 * Design contract: docs/SCAN_DECAY_QWEN36.md. What remains in decode attention
 * after 3a (dequant once per GQA group) and 3b-rot/3b-prefill (no per-row
 * FWHT) is the ROW COUNT: every (kv-head, row) scans all qpos+1 KV rows, O(T)
 * per token. The two-tier scan keeps a deterministic subset:
 *
 *   tier 1 (always):   sink blocks (first Q36_SCAN_SINK rows) + a local
 *                      window (last Q36_SCAN_WIN rows)
 *   tier 2 (gated):    128-row blocks beyond the window, in DESCENDING
 *                      salience, until the cumulative salience share of the
 *                      middle range reaches 1 - Q36_SCAN_EPS
 *
 * Salience is the attention mass a block's rows received, accumulated where
 * scores already exist (flash prefill inner loop, decode settle row) — zero
 * extra passes. THE TABLE IS uint64_t FIXED-POINT (x 2^14), not f32: the
 * flash path runs under schedule(dynamic), so per-thread partial sums group
 * differently run-to-run; integer addition is associative and commutative,
 * so the merged table (and hence the scan set, and hence the generated ids)
 * is bit-reproducible regardless of thread scheduling. Quantising the mass
 * also makes near-tie block rankings stable — float noise cannot flip them.
 *
 /* Amendment to the contract (documented in docs/SCAN_DECAY_QWEN36.md):
  * decode mass accumulates from the SETTLE row (s == 0) only -- the one row
  * of a verify batch that always stands, so no q36_spec snapshot or rollback
  * of the table is needed at all; rejected drafts can never contaminate it.
  * Accepted-draft rows are a deterministic ~(1/(k+1)) subsample of mass
  * events; the ranking is unaffected. Prefill chunks seed ALL their rows --
  * every prompt row stands -- via g_q36_scan_prefill, set by the chunked
  * prefill driver (the only caller with that exact attention() signature),
  * so chunked prefills (E7 progress work) seed completely and verify
  * batches can never reach the flash seeding path.
 *
 * Knobs (defaults preserve today's behavior exactly — Q36_SCAN=full keeps
 * the pre-scan loops verbatim):
 *   Q36_SCAN=full|two-tier   (default full)
 *   Q36_SCAN_WIN=4096  Q36_SCAN_EPS=0.05  Q36_SCAN_SINK=256
 *   Q36_SCAN_DEBUG=1   (top-block dump at the report line)
 * Report: [qwen36 scan] avg scanned rows / total, mass kept.
 */
#ifndef Q36_SCAN_DECAY_H
#define Q36_SCAN_DECAY_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* set only around the chunked prefill driver's attention() call (see the
 * amendment note above): the seeding covers every prefill chunk row and no
 * verify batch can reach it. */
static int g_q36_scan_prefill = 0;

static int q36_scan_two(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("Q36_SCAN");
                 v = (e && e[0] == 't') ? 1 : 0; }   /* "two-tier" */
    return v;
}
static int q36_scan_win(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("Q36_SCAN_WIN"); v = (e && *e) ? atoi(e) : 4096;
                 if (v < 0) v = 0; }
    return v;
}
static int q36_scan_sink(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("Q36_SCAN_SINK"); v = (e && *e) ? atoi(e) : 256;
                 if (v < 0) v = 0; if (v > 4096) v = 4096; }
    return v;
}
static double q36_scan_eps(void) {
    static double v = -1;
    if (v < 0) { const char *e = getenv("Q36_SCAN_EPS"); v = (e && *e) ? atof(e) : 0.05;
                 if (v < 0) v = 0; if (v > 1) v = 1; }
    return v;
}
static int q36_scan_debug(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("Q36_SCAN_DEBUG"); v = (e && *e) ? 1 : 0; }
    return v;
}

#define Q36_SCAN_SCALE 16384.0   /* fixed-point mass: pw in [0,1] -> uint64 */

/* salience cell for (layer, kvh): m->sal is [n_layers][kv_heads][sal_blocks] */
#define Q36_SAL(m, layer, kvh) \
    ((m)->sal + ((int64_t)(layer) * (m)->c.kv_heads + (kvh)) * (m)->sal_blocks)

/* ---- report counters (file-scope; printed once at generation end) -------- */
static uint64_t g_scan_rows, g_scan_total, g_scan_kept, g_scan_mass, g_scan_builds;

/* ---- descending-salience comparator (qsort): sal desc, block asc --------- */
typedef struct { uint64_t s; int b; } Q36ScanPair;
static int q36_scan_cmp(const void *pa, const void *pb) {
    const Q36ScanPair *a = (const Q36ScanPair*)pa, *b = (const Q36ScanPair*)pb;
    if (a->s != b->s) return a->s < b->s ? 1 : -1;   /* descending salience */
    return a->b - b->b;                             /* ties: lower block id */
}

/* Build the scan set for (layer, kvh, qpos): block indices ascending into
 * blks[]. Returns the block count. sel[] must hold sal_blocks bytes (a
 * caller-provided scratch bitmap). Also feeds the report counters. */
static int q36_scan_build(Model *m, int layer, int kvh, int qpos,
                          unsigned char *sel, int *blks)
{
    int nb = m->sal_blocks;
    if (nb <= 0 || qpos < 0) return 0;
    uint64_t *sal = Q36_SAL(m, layer, kvh);
    int bmax = qpos >> 7;                       /* last block with rows <= qpos */
    if (bmax >= nb) bmax = nb - 1;
    memset(sel, 0, (size_t)(bmax + 1));
    int sinkb = q36_scan_sink() >> 7;
    int win   = q36_scan_win();
    int wb0   = (qpos + 1 > win) ? ((qpos + 1 - win) >> 7) : 0;  /* first window blk */

    /* tier 1: sinks + window */
    for (int b = 0; b < sinkb && b <= bmax; b++) sel[b] = 1;
    for (int b = wb0; b <= bmax; b++) sel[b] = 1;

    /* tier 2: middle blocks in descending salience until 1-eps of the
     * middle mass is covered (skipped entirely when eps == 1). */
    double eps = q36_scan_eps();
    if (sinkb < wb0 && eps < 1.0) {
        int M = wb0 - sinkb;
        uint64_t total = 0;
        for (int b = sinkb; b < wb0; b++) total += sal[b];
        if (total > 0) {
            uint64_t target = (uint64_t)((1.0 - eps) * (double)total + 0.5);
            Q36ScanPair *pr = (Q36ScanPair*)malloc((size_t)M * sizeof(Q36ScanPair));
            if (pr) {
                for (int i = 0; i < M; i++) { pr[i].s = sal[sinkb + i]; pr[i].b = sinkb + i; }
                qsort(pr, (size_t)M, sizeof(Q36ScanPair), q36_scan_cmp);
                uint64_t cum = 0;
                for (int i = 0; i < M && cum < target; i++) {
                    cum += pr[i].s; sel[pr[i].b] = 1;
                }
                free(pr);
            }
        }
    }

    int n = 0;
    for (int b = 0; b <= bmax; b++)
        if (sel[b]) {
            blks[n++] = b;
            int r0 = b << 7, r1 = r0 + 128; if (r1 > qpos + 1) r1 = qpos + 1;
            g_scan_rows += (uint64_t)(r1 - r0);
        }
    g_scan_total += (uint64_t)(qpos + 1);
    g_scan_builds++;

    /* mass kept: selected share of the whole-row salience (report only) */
    uint64_t all = 0, kept = 0;
    for (int b = 0; b <= bmax; b++) {
        all += sal[b];
        if (sel[b]) kept += sal[b];
    }
    if (all > 0) { g_scan_kept += kept; g_scan_mass += all; }

    return n;
}

/* [qwen36 scan] report + optional debug dump of layer 0's top blocks. */
static void q36_scan_report(Model *m) {
    if (!q36_scan_two() || !m->sal || g_scan_builds == 0) return;
    double pct = g_scan_total ? 100.0 * (double)g_scan_rows / (double)g_scan_total : 0.0;
    double kept = g_scan_mass ? (double)g_scan_kept / (double)g_scan_mass : 0.0;
    fprintf(stderr, "[qwen36 scan] two-tier: avg scanned rows %llu/%llu (%.1f%%), "
            "salience kept %.4f over %llu builds\n",
            (unsigned long long)g_scan_rows, (unsigned long long)g_scan_total,
            pct, kept, (unsigned long long)g_scan_builds);
    if (q36_scan_debug()) {
        int nb = m->sal_blocks;
        for (int lay = 0; lay < m->c.n_layers; lay++) {
            if (!m->c.is_attn[lay]) continue;   /* DeltaNet layers keep no KV rows */
            for (int kvh = 0; kvh < m->c.kv_heads; kvh++) {
                uint64_t *sal = Q36_SAL(m, lay, kvh);
                fprintf(stderr, "[scan-dbg] L%d kvh%d:", lay, kvh);
                for (int k = 0; k < 4; k++) {
                    int best = -1; uint64_t bs = 0;
                    for (int b = 0; b < nb; b++) if (sal[b] > bs) { bs = sal[b]; best = b; }
                    if (best < 0 || bs == 0) break;
                    fprintf(stderr, " [%d,%d)=%llu", best << 7, (best << 7) + 128,
                            (unsigned long long)bs);
                    sal[best] = 0;   /* destructive top-4 walk (report-time only) */
                }
                fprintf(stderr, "\n");
            }
        }
    }
}

#endif /* Q36_SCAN_DECAY_H */
