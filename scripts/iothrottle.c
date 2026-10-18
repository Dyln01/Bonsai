/* iothrottle.c — LD_PRELOAD I/O fidelity shim (RECONSTRUCTED 2026-10-19).
 *
 * The original harness/iothrottle.c was never committed (docs/REPORT.md §0;
 * ROADMAP item E2). Reconstructed from the REPORT spec:
 *   - interposes open/open64/read/pread/pread64 via dlsym(RTLD_NEXT)
 *   - applies ONLY to files under $IOTH_DIR (the model dir): per-read latency
 *     floor $IOTH_LAT_US + aggregate bandwidth cap $IOTH_BW_MBPS (token-bucket
 *     over a monotonic clock; multithread-safe under one mutex)
 *   - disabled entirely when IOTH_OFF=1; reads of size 0 are free
 *
 * Build:  gcc -O2 -shared -fPIC iothrottle.c -o iothrottle.so -ldl
 * Use:    IOTH_DIR=$MODEL IOTH_LAT_US=100 IOTH_BW_MBPS=150 \
 *           LD_PRELOAD=./iothrottle.so <engine> ...
 * Verified params from REPORT.md: 50 MB/s requested -> 49.4 measured;
 * PCIe4-class (3500, 100us) / SATA-class (150, 500us).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static int    g_on = 0;
static char   g_dir[PATH_MAX];
static size_t g_dir_len = 0;
static long   g_lat_us = 0;
static double g_bw_mbps = 0.0;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static struct timespec g_credit_ts;          /* token-bucket "fully paid up to" */

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void __attribute__((constructor)) ioth_ctor(void) {
    const char *off = getenv("IOTH_OFF");
    if (off && off[0] == '1') return;
    const char *d = getenv("IOTH_DIR");
    if (!d || !*d) return;
    if (!realpath(d, g_dir)) return;
    g_dir_len = strlen(g_dir);
    const char *l = getenv("IOTH_LAT_US");   if (l) g_lat_us  = atol(l);
    const char *b = getenv("IOTH_BW_MBPS");  if (b) g_bw_mbps = atof(b);
    g_on = 1;
    double t = now_s();
    g_credit_ts.tv_sec  = (long)t;
    g_credit_ts.tv_nsec = (long)((t - (double)g_credit_ts.tv_sec) * 1e9);
}

/* fd -> under-managed-dir flag */
#define FD_MAX 4096
static char g_managed[FD_MAX];

static int under_dir(const char *path) {
    char rp[PATH_MAX];
    if (!path || !realpath(path, rp)) return 0;
    return g_on && !strncmp(rp, g_dir, g_dir_len) &&
           (rp[g_dir_len] == '/' || rp[g_dir_len] == '\0');
}

static void pay_bytes(long long n) {
    if (n <= 0) return;
    if (g_lat_us > 0) {
        struct timespec ts = { g_lat_us / 1000000, (g_lat_us % 1000000) * 1000 };
        nanosleep(&ts, NULL);
    }
    if (g_bw_mbps <= 0.0) return;
    /* token bucket: advance the "paid up to" clock by bytes/bw; sleep the debt */
    pthread_mutex_lock(&g_mu);
    double paid = g_credit_ts.tv_sec + 1e-9 * g_credit_ts.tv_nsec;
    double need = (double)n / (g_bw_mbps * 1e6);
    double t = now_s();
    if (paid < t) paid = t;          /* bucket refills while idle */
    paid += need;
    g_credit_ts.tv_sec  = (long)paid;
    g_credit_ts.tv_nsec = (long)((paid - (double)g_credit_ts.tv_sec) * 1e9);
    pthread_mutex_unlock(&g_mu);
    double sleep_for = paid - now_s();
    if (sleep_for > 0) {
        struct timespec ts = { (long)sleep_for, (long)((sleep_for - (double)(long)sleep_for) * 1e9) };
        nanosleep(&ts, NULL);
    }
}

typedef int    (*open_t)(const char *, int, ...);
typedef int    (*open64_t)(const char *, int, ...);
typedef ssize_t (*read_t)(int, void *, size_t);
typedef ssize_t (*pread_t)(int, void *, size_t, off_t);
typedef ssize_t (*pread64_t)(int, void *, size_t, off64_t);

int open(const char *path, int flags, ...) {
    static open_t real = NULL;
    if (!real) real = (open_t)dlsym(RTLD_NEXT, "open");
    mode_t m = 0;
    if (flags & O_CREAT) { __builtin_va_list ap; __builtin_va_start(ap, flags); m = __builtin_va_arg(ap, mode_t); __builtin_va_end(ap); }
    int fd = real(path, flags, m);
    if (g_on && fd >= 0 && fd < FD_MAX) g_managed[fd] = under_dir(path);
    return fd;
}

int open64(const char *path, int flags, ...) {
    static open64_t real = NULL;
    if (!real) real = (open64_t)dlsym(RTLD_NEXT, "open64");
    mode_t m = 0;
    if (flags & O_CREAT) { __builtin_va_list ap; __builtin_va_start(ap, flags); m = __builtin_va_arg(ap, mode_t); __builtin_va_end(ap); }
    int fd = real(path, flags, m);
    if (g_on && fd >= 0 && fd < FD_MAX) g_managed[fd] = under_dir(path);
    return fd;
}

ssize_t read(int fd, void *buf, size_t n) {
    static read_t real = NULL;
    if (!real) real = (read_t)dlsym(RTLD_NEXT, "read");
    ssize_t r = real(fd, buf, n);
    if (r > 0 && g_on && fd >= 0 && fd < FD_MAX && g_managed[fd]) pay_bytes(r);
    return r;
}

ssize_t pread(int fd, void *buf, size_t n, off_t off) {
    static pread_t real = NULL;
    if (!real) real = (pread_t)dlsym(RTLD_NEXT, "pread");
    ssize_t r = real(fd, buf, n, off);
    if (r > 0 && g_on && fd >= 0 && fd < FD_MAX && g_managed[fd]) pay_bytes(r);
    return r;
}

ssize_t pread64(int fd, void *buf, size_t n, off64_t off) {
    static pread64_t real = NULL;
    if (!real) real = (pread64_t)dlsym(RTLD_NEXT, "pread64");
    ssize_t r = real(fd, buf, n, off);
    if (r > 0 && g_on && fd >= 0 && fd < FD_MAX && g_managed[fd]) pay_bytes(r);
    return r;
}
