/* fbm-gpu: the OpenCL (GPU) front end. It has no CPU-specific code, so it
 * builds wherever there is a C11 compiler and an OpenCL driver, Windows
 * included. */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "blocks.h"
#include "gpu.h"
#include "header.h"
#include "kernels.h"
#include "sha256.h"
#include "stats.h"
#include "util.h"

#ifndef FBM_VERSION
#define FBM_VERSION "unknown"
#endif

/* Layouts 0 and 1 are fbm_gpu_layout; 2 is the optional cgminer baseline. */
enum { POCLBM = 2 };
static const char *const layout_name[] = {"nonce", "vr", "poclbm"};
static int have_baseline;

static void usage(void)
{
    fprintf(stderr,
            "usage:\n"
            "  fbm-gpu list                      OpenCL platforms and devices\n"
            "  fbm-gpu info                      device and compiled-kernel resources\n"
            "  fbm-gpu test                      correctness tests against the CPU reference\n"
            "  fbm-gpu bench [options]           interleaved hash-rate benchmark\n"
            "      --kernels nonce,vr,poclbm (default all available)  --rounds R (default 10)\n"
            "      --seconds S per run (default 2)     --versions N for vr (default 65536)\n"
            "  fbm-gpu mine --header HEX80 [options] search for valid (version, nonce) pairs\n"
            "      --start NONCE --count N (default: whole 2^32 range)\n"
            "      --versions N (BIP 320 rolled versions, default 1; 64+ uses vr)\n"
            "      --kernel nonce|vr   --seconds S (stop after S seconds; Ctrl-C also stops)\n"
            "  fbm-gpu dump --out FILE           save the driver-compiled kernels (AMD: ELF,\n"
            "                                    for tools/gpu_isacheck.py)\n"
            "  fbm-gpu probe                     per-instruction issue rates (GPU clock probe)\n"
            "  fbm-gpu sustain [--kernel vr|nonce] [--seconds S]  long run, logs rate and, on\n"
            "                                    Linux, board power and clock (J/TH)\n"
            "  fbm-gpu report [--out FILE] [--sustain S]  everything above in one file, to\n"
            "                                    send back (default: fbm-gpu-report.txt)\n"
            "common options:\n"
            "  --platform P --device D           pick a device (default: first GPU)\n"
            "  --wg N                            work-group size (default 64)\n"
            "  --nonce-iters N --vr-iters N      nonces per work-item per launch (16, 64)\n"
            "  --launch-ms MS                    target kernel launch length (default 8, which\n"
            "                                    keeps a desktop on the same GPU responsive)\n"
            "  --dedicated                       100 ms launches, for a GPU with no display\n"
            "  --program FILE                    run a prebuilt program binary instead of the\n"
            "                                    source (make gpu-codeobj: the audited ISA)\n"
            "  --baseline FILE                   cgminer's poclbm kernel, run as the prior-art\n"
            "                                    baseline (default: bench/gpu-baselines/src/\n"
            "                                    poclbm130302.cl, from fetch.sh, if present)\n");
}

static const char *arg_value(int argc, char **argv, const char *name, const char *def)
{
    for (int i = 0; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    }
    return def;
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

static double now(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
static double now(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

static int has_flag(int argc, char **argv, const char *name)
{
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], name) == 0)
            return 1;
    }
    return 0;
}

static fbm_gpu *open_from_args(int argc, char **argv)
{
    fbm_gpu_opts o;
    fbm_gpu_default_opts(&o);
    o.platform = atoi(arg_value(argc, argv, "--platform", "-1"));
    o.device = atoi(arg_value(argc, argv, "--device", "-1"));
    o.wg = (unsigned)atoi(arg_value(argc, argv, "--wg", "64"));
    o.nonce_iters = (unsigned)atoi(arg_value(argc, argv, "--nonce-iters", "16"));
    o.vr_iters = (unsigned)atoi(arg_value(argc, argv, "--vr-iters", "64"));
    o.program = arg_value(argc, argv, "--program", NULL);
    o.launch_ms = atof(arg_value(argc, argv, "--launch-ms",
                                 has_flag(argc, argv, "--dedicated") ? "100" : "8"));
    if (o.wg == 0 || o.wg > 1024 || o.nonce_iters == 0 || o.vr_iters == 0 ||
        o.launch_ms <= 0 || o.launch_ms > 1000) {
        fprintf(stderr, "bad tuning option\n");
        return NULL;
    }
    fbm_gpu *g = fbm_gpu_open(&o);
    if (!g)
        return NULL;
    printf("device: %s\n", fbm_gpu_name(g));
    /* The prior-art baseline, if fetched (bench/gpu-baselines/fetch.sh). */
    const char *bpath = arg_value(argc, argv, "--baseline",
                                  "bench/gpu-baselines/src/poclbm130302.cl");
    have_baseline = fbm_gpu_load_baseline(g, bpath) == 0;
    if (!have_baseline && has_flag(argc, argv, "--baseline"))
        fprintf(stderr, "warning: baseline %s missing or does not build\n", bpath);
    return g;
}

/* ---- tests ------------------------------------------------------------- */

static int failures;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            failures++;                                   \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
        }                                                 \
    } while (0)

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint32_t rng32(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545f4914f6cdd1dull) >> 32);
}

static int hit_cmp(const void *a, const void *b)
{
    const fbm_hit *x = a, *y = b;
    if (x->version != y->version)
        return x->version < y->version ? -1 : 1;
    if (x->nonce != y->nonce)
        return x->nonce < y->nonce ? -1 : 1;
    return 0;
}

static fbm_hits new_hits(size_t cap)
{
    fbm_hits h = {calloc(cap, sizeof(fbm_hit)), cap, 0};
    return h;
}

static void sort_hits(fbm_hits *h)
{
    qsort(h->v, h->n < h->cap ? h->n : h->cap, sizeof(fbm_hit), hit_cmp);
}

/* The oracle: the portable reference SHA-256d over the whole header. */
static fbm_hits ref_scan(const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0, uint64_t nn)
{
    fbm_hits h = new_hits(1u << 16);
    const uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);
    uint8_t hdr[80], hash[32];
    memcpy(hdr, job->header, 80);
    for (uint32_t r = r0; r < r0 + nr; r++) {
        const uint32_t version = fbm_rolled_version(base, r);
        fbm_header_set(hdr, FBM_OFF_VERSION, version);
        for (uint64_t i = 0; i < nn; i++) {
            fbm_header_set(hdr, FBM_OFF_NONCE, (uint32_t)(n0 + i));
            fbm_sha256d(hdr, 80, hash);
            if (fbm_top32_le(hash) <= job->t7)
                fbm_hits_push(&h, version, (uint32_t)(n0 + i));
        }
    }
    sort_hits(&h);
    return h;
}

static void diff_case(fbm_gpu *g, const char *label, const fbm_job *job, uint32_t r0, uint32_t nr,
                      uint32_t n0, uint64_t nn)
{
    fbm_hits want = ref_scan(job, r0, nr, n0, nn);
    CHECK(want.n <= want.cap, "%s: too many reference hits", label);
    for (int layout = 0; layout < 2; layout++) {
        fbm_hits got = new_hits(1u << 16);
        uint64_t done = fbm_gpu_scan(g, (fbm_gpu_layout)layout, job, r0, nr, n0, nn, &got);
        sort_hits(&got);
        CHECK(done == (uint64_t)nr * nn, "%s/%s: returned %llu hashes", label, layout_name[layout],
              (unsigned long long)done);
        int same = got.n == want.n && !memcmp(got.v, want.v, want.n * sizeof(fbm_hit));
        CHECK(same, "%s/%s: %zu candidates, reference %zu", label, layout_name[layout], got.n,
              want.n);
        free(got.v);
    }
    printf("  %-38s %zu candidates\n", label, want.n);
    free(want.v);
}

static void random_job(fbm_job *job, uint32_t t7)
{
    for (int i = 0; i < 80; i++)
        job->header[i] = (uint8_t)rng32();
    job->t7 = t7;
}

static void test_differential(fbm_gpu *g, const char *tag)
{
    fbm_job job;
    char label[64];
    printf("GPU kernels vs reference, %s\n", tag);
#define CASE(name, t7, r0, nr, n0, nn)                          \
    do {                                                       \
        random_job(&job, t7);                                  \
        snprintf(label, sizeof label, "%s", name);             \
        diff_case(g, label, &job, r0, nr, n0, nn);             \
    } while (0)
    CASE("loose (~1/16)", 0x0fffffff, 0, 32, 1000, 4096);
    CASE("tight (~1/4096)", 0x000fffff, 7, 48, 123456789, 16384);
    CASE("nearly all", 0xfffffffe, 0, 16, 0, 64);
    CASE("odd sizes", 0x3fffffff, 3, 5, 77, 37);
    CASE("end of nonce and version spaces", 0x3fffffff, FBM_VR_MAX - 17, 17, 0xffffffffu - 100,
         101);
    CASE("production t7=0", 0, 0, 16, 0, 1 << 14);
    CASE("many versions, not a group multiple", 0x00ffffff, FBM_VR_MAX - 1000, 1000, 5, 300);
#undef CASE
}

/* Every layout must rediscover the real (version, nonce) of real blocks. The
 * base version is offset so that the answer sits at rolled index r = 5. */
static void test_known(fbm_gpu *g)
{
    printf("GPU kernels rediscover real (version, nonce) of %zu mainnet blocks\n",
           fbm_known_block_count);
    for (size_t b = 0; b < fbm_known_block_count; b++) {
        fbm_job job;
        uint8_t target[32], hash[32];
        fbm_hex_decode(fbm_known_blocks[b].hex, job.header, 80);
        fbm_bits_to_target(fbm_header_get(job.header, FBM_OFF_BITS), target);
        job.t7 = fbm_top32_le(target);
        const uint32_t nonce = fbm_header_get(job.header, FBM_OFF_NONCE);
        const uint32_t version = fbm_header_get(job.header, FBM_OFF_VERSION);
        fbm_header_set(job.header, FBM_OFF_VERSION, fbm_rolled_version(version, 5));
        for (int layout = 0; layout < 2; layout++) {
            fbm_hits h = new_hits(4096);
            fbm_gpu_scan(g, (fbm_gpu_layout)layout, &job, 0, 16, nonce - 3000, 5000, &h);
            int found = 0;
            for (size_t j = 0; j < h.n && j < h.cap; j++) {
                uint8_t hdr[80];
                memcpy(hdr, job.header, 80);
                fbm_header_set(hdr, FBM_OFF_VERSION, h.v[j].version);
                fbm_header_set(hdr, FBM_OFF_NONCE, h.v[j].nonce);
                fbm_sha256d(hdr, 80, hash);
                CHECK(fbm_top32_le(hash) <= job.t7, "%s/%s: bogus candidate",
                      fbm_known_blocks[b].name, layout_name[layout]);
                if (h.v[j].nonce == nonce && h.v[j].version == version)
                    found = 1;
            }
            CHECK(found, "%s/%s: missed version 0x%08x nonce %u", fbm_known_blocks[b].name,
                  layout_name[layout], version, nonce);
            free(h.v);
        }
    }
}

/* A full device buffer must be counted, never silently truncated. */
static void test_overflow(fbm_gpu *g)
{
    fbm_gpu_opts *o = fbm_gpu_options(g);
    const uint32_t saved = o->hit_cap;
    fbm_job job;
    printf("candidate overflow is counted\n");
    random_job(&job, 0xffffffff); /* every hash is a candidate */
    o->hit_cap = 100;
    for (int layout = 0; layout < 2; layout++) {
        fbm_hits h = new_hits(4096);
        fbm_gpu_scan(g, (fbm_gpu_layout)layout, &job, 0, 16, 1000, 64, &h);
        CHECK(h.n == 16 * 64, "%s: counted %zu of 1024 candidates", layout_name[layout], h.n);
        for (size_t j = 0; j < h.n && j < 100; j++)
            CHECK(h.v[j].nonce >= 1000 && h.v[j].nonce < 1064, "%s: nonce %u out of range",
                  layout_name[layout], h.v[j].nonce);
        free(h.v);
    }
    o->hit_cap = saved;
}

/* A small scan with a loose threshold, checked exactly against the host
 * reference. On a card that is unstable (overclock, undervolt, heat) or a
 * driver that miscompiles, this fails; the host re-check of candidates only
 * catches false positives, never missed blocks. */
static int canary(fbm_gpu *g, fbm_gpu_layout layout)
{
    fbm_job job;
    random_job(&job, 0x00ffffff); /* ~1/256 of hashes are candidates */
    const uint32_t nr = layout == FBM_GPU_VR ? 64 : 1, n0 = rng32() & 0x7fffffff;
    const uint64_t nn = layout == FBM_GPU_VR ? 256 : 16384;
    fbm_hits want = ref_scan(&job, 0, nr, n0, nn), got = new_hits(1u << 16);
    fbm_gpu_scan(g, layout, &job, 0, nr, n0, nn, &got);
    sort_hits(&got);
    const int ok = got.n == want.n && !memcmp(got.v, want.v, want.n * sizeof(fbm_hit));
    free(got.v);
    free(want.v);
    return ok;
}

static void hardware_error(const char *what)
{
    fprintf(stderr, "\nERROR: %s.\nThe kernels pass their tests, so this points at the GPU or its driver: "
            "if you changed clocks, voltage or the power limit, undo that and run `fbm-gpu test`. "
            "Stopping, because a card that computes wrong hashes can also miss blocks.\n", what);
}

/* The canary must notice a device that computes wrong hashes. */
static void test_canary(fbm_gpu *g)
{
    fbm_gpu_opts *o = fbm_gpu_options(g);
    printf("the mining canary detects a faulty device\n");
    for (int layout = 0; layout < 2; layout++) {
        CHECK(canary(g, (fbm_gpu_layout)layout), "%s: canary fails on a good device",
              layout_name[layout]);
        o->inject_fault = 1;
        CHECK(!canary(g, (fbm_gpu_layout)layout), "%s: canary misses an injected fault",
              layout_name[layout]);
        o->inject_fault = 0;
    }
}

/* Production geometry: the measured launch sizes (several launches per
 * scan), the answer at the highest version index and at the last nonce of
 * the range, so the last work-group of the last launch must find it. */
static void test_planted(fbm_gpu *g)
{
    printf("planted answers at production geometry (last version index, last nonce)\n");
    for (int layout = 0; layout < 2; layout++) {
        /* genesis for the nonce layout (a large nonce leaves room below it),
         * block 800000 (rolled version) for the version layout */
        const fbm_known_block *b = &fbm_known_blocks[layout == FBM_GPU_VR ? 2 : 0];
        fbm_job job;
        uint8_t target[32];
        fbm_hex_decode(b->hex, job.header, 80);
        fbm_bits_to_target(fbm_header_get(job.header, FBM_OFF_BITS), target);
        job.t7 = fbm_top32_le(target);
        const uint32_t nonce = fbm_header_get(job.header, FBM_OFF_NONCE);
        const uint32_t version = fbm_header_get(job.header, FBM_OFF_VERSION);
        const uint32_t top = FBM_VR_MAX - 1;
        fbm_header_set(job.header, FBM_OFF_VERSION, fbm_rolled_version(version, top));
        /* ~0.3 s of work at the measured rate, at least a few launches */
        double want = fbm_gpu_rate(g, (fbm_gpu_layout)layout) * 0.3;
        uint32_t r0 = layout == FBM_GPU_VR ? 0 : top, nr = layout == FBM_GPU_VR ? FBM_VR_MAX : 1;
        uint64_t nn = (uint64_t)(want / nr);
        const uint64_t lo = layout == FBM_GPU_VR ? 64 : 1u << 20;
        const uint64_t hi = layout == FBM_GPU_VR ? 4096 : 1u << 30;
        nn = nn < lo ? lo : nn > hi ? hi : nn;
        nn = nn > (uint64_t)nonce + 1 ? (uint64_t)nonce + 1 : nn;
        fbm_hits h = new_hits(4096);
        fbm_gpu_scan(g, (fbm_gpu_layout)layout, &job, r0, nr, nonce - (uint32_t)(nn - 1), nn, &h);
        int found = 0;
        for (size_t j = 0; j < h.n && j < h.cap; j++)
            found |= h.v[j].nonce == nonce && h.v[j].version == version;
        CHECK(found, "%s: missed the answer at version index %u, nonce %u (last of %llu)",
              layout_name[layout], top, nonce, (unsigned long long)nn);
        printf("  %-5s %s: %u versions x %llu nonces, answer found: %s\n", layout_name[layout],
               b->name, nr, (unsigned long long)nn, found ? "yes" : "NO");
        free(h.v);
    }
}

/* The ported baseline must find the real nonce of every known block (with
 * its true version: it cannot roll) and report nothing false. */
static void test_baseline(fbm_gpu *g)
{
    printf("baseline poclbm rediscovers the real nonce of %zu mainnet blocks\n",
           fbm_known_block_count);
    for (size_t b = 0; b < fbm_known_block_count; b++) {
        fbm_job job;
        uint8_t target[32], hash[32];
        fbm_hex_decode(fbm_known_blocks[b].hex, job.header, 80);
        fbm_bits_to_target(fbm_header_get(job.header, FBM_OFF_BITS), target);
        job.t7 = fbm_top32_le(target);
        const uint32_t nonce = fbm_header_get(job.header, FBM_OFF_NONCE);
        fbm_hits h = new_hits(64);
        fbm_gpu_scan_baseline(g, &job, 0, 1, fbm_bswap32(nonce) - 3000, 5000, &h);
        int found = 0;
        for (size_t j = 0; j < h.n && j < h.cap; j++) {
            uint8_t hdr[80];
            memcpy(hdr, job.header, 80);
            fbm_header_set(hdr, FBM_OFF_NONCE, h.v[j].nonce);
            fbm_sha256d(hdr, 80, hash);
            CHECK(fbm_top32_le(hash) <= job.t7, "%s/poclbm: bogus candidate", fbm_known_blocks[b].name);
            found |= h.v[j].nonce == nonce;
        }
        CHECK(found, "%s/poclbm: missed nonce %u", fbm_known_blocks[b].name, nonce);
        free(h.v);
    }
}

/* Every test; returns the number of failures. */
static int run_tests(fbm_gpu *g)
{
    fbm_gpu_opts *o = fbm_gpu_options(g);
    failures = 0;
    test_differential(g, "default launch sizes");
    /* Tiny launches: many launches per scan, partial work-groups, iteration
     * counts that do not divide the range. */
    const fbm_gpu_opts saved = *o;
    o->launch_hashes = 1000; /* fixed, instead of launch_ms */
    o->nonce_iters = 3;
    o->vr_iters = 7;
    test_differential(g, "tiny launches (1000 hashes, 3 and 7 nonces per work-item)");
    *o = saved;
    test_known(g);
    test_planted(g);
    test_overflow(g);
    test_canary(g);
    if (have_baseline)
        test_baseline(g);
    else
        printf("(baseline poclbm not loaded: run bench/gpu-baselines/fetch.sh to include it)\n");
    printf(failures ? "%d FAILURE(S)\n" : "all GPU tests passed\n", failures);
    return failures;
}

static int cmd_test(int argc, char **argv)
{
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g)
        return 2;
    const int f = run_tests(g);
    fbm_gpu_close(g);
    return f ? 1 : 0;
}

/* ---- bench --------------------------------------------------------------- */

static volatile sig_atomic_t stop_requested;

static void on_sigint(int sig)
{
    (void)sig;
    stop_requested = 1;
}

#define MAX_ROUNDS 200

typedef struct {
    int layout; /* fbm_gpu_layout, or POCLBM */
    uint32_t nr;
    uint64_t nn;
    double rate[MAX_ROUNDS];  /* wall clock, MH/s */
    double krate[MAX_ROUNDS]; /* device kernel time, MH/s */
} bench_entry;

static double bench_once(fbm_gpu *g, bench_entry *e, const fbm_job *job, double *krate)
{
    fbm_hit buf[64];
    fbm_hits hits = {buf, 64, 0};
    const double t0 = now();
    uint64_t n = e->layout == POCLBM ? fbm_gpu_scan_baseline(g, job, 0, 1, 0, e->nn, &hits)
                                     : fbm_gpu_scan(g, (fbm_gpu_layout)e->layout, job, 0, e->nr,
                                                    0, e->nn, &hits);
    const double dt = now() - t0;
    *krate = n / fbm_gpu_kernel_seconds(g) / 1e6;
    return n / dt / 1e6;
}

/* Interleaved rounds, each running every selected layout once in rotating
 * order; the vr / nonce ratio is a median of per-round ratios with a
 * bootstrap 95% CI. */
static int run_bench(fbm_gpu *g, const char *which, int rounds, double seconds, uint32_t versions)
{
    bench_entry es[3];
    int ne = 0;
    fbm_job job;

    if (strstr(which, "poclbm") && have_baseline)
        es[ne++] = (bench_entry){.layout = POCLBM, .nr = 1};
    if (strstr(which, "nonce"))
        es[ne++] = (bench_entry){.layout = FBM_GPU_NONCE, .nr = 1};
    if (strstr(which, "vr"))
        es[ne++] = (bench_entry){.layout = FBM_GPU_VR, .nr = versions};
    if (!ne)
        return -1;
    /* Speedups are against the prior-art kernel if it runs, else the
     * nonce layout. */
    const int base = 0;

    /* A fixed pseudo-random header; t7 = 0 as for any real network target. */
    for (int i = 0; i < 80; i++)
        job.header[i] = (uint8_t)(i * 37 + 11);
    job.t7 = 0;

    /* Calibrate each layout to ~`seconds` per run (doubles as warm-up). */
    for (int i = 0; i < ne; i++) {
        es[i].nn = 64;
        for (;;) {
            double kr;
            const uint64_t hashes = (uint64_t)es[i].nr * es[i].nn;
            const double rate = bench_once(g, &es[i], &job, &kr);
            const double dt = hashes / (rate * 1e6);
            if (dt > 0.25 || es[i].nn >= (1ull << 32) / 4) {
                double nn = es[i].nn * seconds / dt;
                es[i].nn = nn > 4294967295.0 ? 4294967295ull : nn < 64 ? 64 : (uint64_t)nn;
                break;
            }
            es[i].nn *= 4;
        }
    }
    printf("%d interleaved rounds of ~%.1f s per kernel; vr rolls %u versions per nonce\n",
           rounds, seconds, versions);
    for (int r = 0; r < rounds; r++) {
        for (int j = 0; j < ne; j++) {
            bench_entry *e = &es[(j + r) % ne];
            e->rate[r] = bench_once(g, e, &job, &e->krate[r]);
        }
        fprintf(stderr, "\rround %d/%d", r + 1, rounds);
    }
    fprintf(stderr, "\r                \r");

    printf("| kernel | MH/s median (wall) | IQR | min..max | MH/s (kernel time) | vs %s [95%% CI] |\n"
           "|---|---:|---:|---:|---:|---:|\n", layout_name[es[base].layout]);
    for (int i = 0; i < ne; i++) {
        double s[MAX_ROUNDS], ratio[MAX_ROUNDS], lo, hi;
        memcpy(s, es[i].rate, sizeof(double) * rounds);
        qsort(s, rounds, sizeof(double), fbm_cmp_double);
        for (int r = 0; r < rounds; r++)
            ratio[r] = es[i].rate[r] / es[base].rate[r];
        fbm_bootstrap_ci(ratio, rounds, &lo, &hi);
        printf("| %s%s | %.1f | %.1f..%.1f | %.1f..%.1f | %.1f | %.3fx [%.3f, %.3f] |\n",
               layout_name[es[i].layout], es[i].layout == POCLBM ? " (cgminer 3.7.2)" : "",
               fbm_quantile(s, rounds, 0.5), fbm_quantile(s, rounds, 0.25),
               fbm_quantile(s, rounds, 0.75), s[0], s[rounds - 1],
               fbm_median(es[i].krate, rounds), fbm_median(ratio, rounds), lo, hi);
    }
    printf("\nwall = scan time on the host clock, including launch gaps and the schedule\n"
           "pre-pass; kernel time = sum of OpenCL profiling intervals.\n");
    return 0;
}

static int cmd_bench(int argc, char **argv)
{
    const char *which = arg_value(argc, argv, "--kernels", "poclbm,nonce,vr");
    int rounds = atoi(arg_value(argc, argv, "--rounds", "10"));
    double seconds = atof(arg_value(argc, argv, "--seconds", "2"));
    uint32_t versions = (uint32_t)strtoul(arg_value(argc, argv, "--versions", "65536"), NULL, 0);
    if (rounds < 1 || rounds > MAX_ROUNDS || seconds <= 0 || versions == 0 ||
        versions > FBM_VR_MAX) {
        usage();
        return 2;
    }
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g)
        return 2;
    const int rc = run_bench(g, which, rounds, seconds, versions);
    fbm_gpu_close(g);
    if (rc < 0)
        usage();
    return rc < 0 ? 2 : 0;
}

/* ---- sustained runs, power --------------------------------------------- */

/* Board power (W) and shader clock (MHz) of the first AMD GPU, from Linux
 * hwmon; returns 0 where that is not available (other OSes). */
static int power_read(double *watts, double *mhz)
{
    char path[160];
    for (int card = 0; card < 8; card++) {
        unsigned vendor = 0;
        snprintf(path, sizeof path, "/sys/class/drm/card%d/device/vendor", card);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        int ok = fscanf(f, "%x", &vendor) == 1;
        fclose(f);
        if (!ok || vendor != 0x1002)
            continue;
        for (int hw = 0; hw < 16; hw++) {
            static const char *const pw[] = {"power1_average", "power1_input"};
            double uw = -1, hz = -1;
            for (int k = 0; k < 2 && uw < 0; k++) {
                snprintf(path, sizeof path, "/sys/class/drm/card%d/device/hwmon/hwmon%d/%s",
                         card, hw, pw[k]);
                if ((f = fopen(path, "r"))) {
                    if (fscanf(f, "%lf", &uw) != 1)
                        uw = -1;
                    fclose(f);
                }
            }
            if (uw < 0)
                continue;
            snprintf(path, sizeof path, "/sys/class/drm/card%d/device/hwmon/hwmon%d/freq1_input",
                     card, hw);
            if ((f = fopen(path, "r"))) {
                if (fscanf(f, "%lf", &hz) != 1)
                    hz = -1;
                fclose(f);
            }
            *watts = uw * 1e-6;
            *mhz = hz > 0 ? hz * 1e-6 : 0;
            return 1;
        }
    }
    return 0;
}

/* Mines a fixed job for `seconds`, printing the rate (and power) every 10 s.
 * Returns the average rate in H/s. */
static double run_sustain(fbm_gpu *g, fbm_gpu_layout layout, double seconds, uint32_t versions)
{
    fbm_job job;
    for (int i = 0; i < 80; i++)
        job.header[i] = (uint8_t)(i * 29 + 3);
    job.t7 = 0;
    const uint32_t nr = layout == FBM_GPU_VR ? versions : 1;
    double w, mhz;
    const int have_power = power_read(&w, &mhz);
    printf("sustained %s run, %.0f s%s\n", layout_name[layout], seconds,
           have_power ? "" : "; board power is not readable here (Linux hwmon only): read it "
                             "from Adrenalin's metrics overlay or HWiNFO and note it");
    printf("| t (s) | MH/s | board W | shader MHz | J/TH |\n|---:|---:|---:|---:|---:|\n");
    const double t0 = now();
    double last = t0, win_h = 0, win_w = 0, all_h = 0, all_w = 0;
    int win_n = 0, all_n = 0;
    uint32_t n0 = 0;
    while (now() - t0 < seconds && !stop_requested) {
        uint64_t nn = (uint64_t)(fbm_gpu_rate(g, layout) * 0.5 / nr);
        nn = nn < 1 ? 1 : nn > (1ull << 32) - n0 ? (1ull << 32) - n0 : nn;
        fbm_hit buf[64];
        fbm_hits hits = {buf, 64, 0};
        win_h += fbm_gpu_scan(g, layout, &job, 0, nr, n0, nn, &hits);
        n0 = (uint32_t)(n0 + nn); /* wraps: a new region, same cost */
        if (have_power && power_read(&w, &mhz)) {
            win_w += w;
            win_n++;
        }
        const double t = now();
        if (t - last >= 10 || t - t0 >= seconds) {
            const double rate = win_h / (t - last);
            if (win_n) {
                const double pw = win_w / win_n;
                printf("| %.0f | %.1f | %.1f | %.0f | %.0f |\n", t - t0, rate / 1e6, pw, mhz,
                       pw / (rate / 1e12));
            } else {
                printf("| %.0f | %.1f | - | - | - |\n", t - t0, rate / 1e6);
            }
            fflush(stdout);
            all_h += win_h;
            all_w += win_w;
            all_n += win_n;
            win_h = win_w = 0;
            win_n = 0;
            last = t;
        }
    }
    const double dt = now() - t0, rate = (all_h + win_h) / dt;
    printf("average %.1f MH/s over %.0f s", rate / 1e6, dt);
    if (all_n)
        printf(", %.1f W, %.0f J/TH", all_w / all_n, all_w / all_n / (rate / 1e12));
    printf("\n");
    return rate;
}

static int cmd_sustain(int argc, char **argv)
{
    const char *k = arg_value(argc, argv, "--kernel", "vr");
    const double seconds = atof(arg_value(argc, argv, "--seconds", "300"));
    const uint32_t versions = (uint32_t)strtoul(arg_value(argc, argv, "--versions", "65536"),
                                                NULL, 0);
    if (seconds <= 0 || versions == 0 || versions > FBM_VR_MAX) {
        usage();
        return 2;
    }
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g)
        return 2;
    signal(SIGINT, on_sigint);
    run_sustain(g, strcmp(k, "nonce") == 0 ? FBM_GPU_NONCE : FBM_GPU_VR, seconds, versions);
    fbm_gpu_close(g);
    return 0;
}

/* ---- mine ---------------------------------------------------------------- */

static int cmd_mine(int argc, char **argv)
{
    const char *hex = arg_value(argc, argv, "--header", NULL);
    const char *kname = arg_value(argc, argv, "--kernel", NULL);
    uint64_t start = strtoull(arg_value(argc, argv, "--start", "0"), NULL, 0);
    uint64_t count = strtoull(arg_value(argc, argv, "--count", "4294967296"), NULL, 0);
    uint32_t versions = (uint32_t)strtoul(arg_value(argc, argv, "--versions", "1"), NULL, 0);
    const double limit = atof(arg_value(argc, argv, "--seconds", "0"));
    fbm_job job;
    uint8_t target[32];

    if (!hex || fbm_hex_decode(hex, job.header, 80) != 0) {
        fprintf(stderr, "need --header with 160 hex characters\n");
        return 2;
    }
    if (fbm_bits_to_target(fbm_header_get(job.header, FBM_OFF_BITS), target) != 0) {
        fprintf(stderr, "invalid compact target in header\n");
        return 2;
    }
    if (start > 0xffffffffull || count == 0 || start + count > (1ull << 32) || versions == 0 ||
        versions > FBM_VR_MAX) {
        fprintf(stderr, "bad nonce or version range\n");
        return 2;
    }
    fbm_gpu_layout layout = versions >= 64 ? FBM_GPU_VR : FBM_GPU_NONCE;
    if (kname)
        layout = strcmp(kname, "vr") == 0 ? FBM_GPU_VR : FBM_GPU_NONCE;
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g)
        return 2;
    job.t7 = fbm_top32_le(target);
    printf("kernel %s, difficulty %.3f, %u version(s) x %llu nonces from %llu\n",
           layout_name[layout], fbm_target_difficulty(target), versions,
           (unsigned long long)count, (unsigned long long)start);
    if (!canary(g, layout)) {
        hardware_error("the start-up check (a known candidate set) came back wrong");
        fbm_gpu_close(g);
        return 3;
    }
    signal(SIGINT, on_sigint);

    int solutions = 0, rc = 1;
    size_t candidates = 0;
    uint64_t hashes = 0, off = 0;
    const double t0 = now();
    double next_canary = t0 + 60;
    for (int slice = 0; off < count && !stop_requested && !(limit > 0 && now() - t0 >= limit);
         slice++) {
        /* Slices of ~1 s of work, so Ctrl-C, --seconds and progress respond;
         * the first is short, until the rate has been measured. */
        uint64_t nn = (uint64_t)(fbm_gpu_rate(g, layout) * (slice ? 1.0 : 0.1) / versions);
        nn = nn < 1 ? 1 : nn > count - off ? count - off : nn;
        fbm_hit buf[4096];
        fbm_hits hits = {buf, 4096, 0};
        hashes += fbm_gpu_scan(g, layout, &job, 0, versions, (uint32_t)(start + off), nn, &hits);
        off += nn;
        candidates += hits.n;
        for (size_t i = 0; i < hits.n && i < hits.cap; i++) {
            uint8_t hdr[80], hash[32];
            char disp[65], full[161];
            memcpy(hdr, job.header, 80);
            fbm_header_set(hdr, FBM_OFF_VERSION, hits.v[i].version);
            fbm_header_set(hdr, FBM_OFF_NONCE, hits.v[i].nonce);
            fbm_sha256d(hdr, 80, hash);
            if (fbm_top32_le(hash) > job.t7) {
                /* The filter is exact, so this is a wrong hash on the GPU. */
                char what[160];
                snprintf(what, sizeof what, "the GPU reported version 0x%08x nonce %u, which "
                         "fails its own pre-filter", hits.v[i].version, hits.v[i].nonce);
                hardware_error(what);
                fbm_gpu_close(g);
                return 3;
            }
            if (fbm_cmp256_le(hash, target) > 0)
                continue;
            solutions++;
            fbm_hash_display(hash, disp);
            fbm_hex_encode(hdr, 80, full);
            printf("SOLUTION version=0x%08x nonce=%u hash=%s\n  header=%s\n", hits.v[i].version,
                   hits.v[i].nonce, disp, full);
        }
        if (hits.n > hits.cap)
            fprintf(stderr, "warning: %zu candidates in one slice, only %zu checked\n", hits.n,
                    hits.cap);
        if (now() >= next_canary) {
            if (!canary(g, layout)) {
                hardware_error("the periodic check (a known candidate set) came back wrong");
                fbm_gpu_close(g);
                return 3;
            }
            next_canary = now() + 60;
        }
        const double dt = now() - t0;
        fprintf(stderr, "\r%.2f%% of the range, %.1f MH/s   ", 100.0 * off / count,
                hashes / dt / 1e6);
    }
    const double dt = now() - t0;
    fprintf(stderr, "\n");
    if (stop_requested || off < count)
        printf("stopped at nonce offset %llu of %llu%s\n", (unsigned long long)off,
               (unsigned long long)count, stop_requested ? " (Ctrl-C)" : " (--seconds)");
    printf("%llu hashes in %.3f s = %.3f MH/s, %d solution(s), %zu candidate(s)\n",
           (unsigned long long)hashes, dt, hashes / dt / 1e6, solutions, candidates);
    rc = solutions ? 0 : 1;
    fbm_gpu_close(g);
    return rc;
}

/* ---- report ------------------------------------------------------------ */

static int cmd_probe(int argc, char **argv)
{
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g)
        return 2;
    const int rc = fbm_gpu_probe(g, stdout);
    fbm_gpu_close(g);
    return rc ? 1 : 0;
}

/* One command for the card's owner: everything this VM could not measure,
 * in one file to send back. */
static int cmd_report(int argc, char **argv)
{
    const char *path = arg_value(argc, argv, "--out", "fbm-gpu-report.txt");
    const double sus = atof(arg_value(argc, argv, "--sustain", "300"));
    const time_t t = time(NULL);

    if (sus < 0) {
        usage();
        return 2;
    }
    fprintf(stderr, "writing %s; this takes about %.0f minutes. Close games and videos: the GPU "
            "should be otherwise idle.\n", path, (2 * sus + 150) / 60);
    if (!freopen(path, "w", stdout)) {
        perror(path);
        return 2;
    }
    printf("# fbm-gpu report\n\nfbm-gpu %s, %s\n## OpenCL devices\n\n```\n", FBM_VERSION,
           ctime(&t));
    fbm_gpu_list();
    printf("```\n\n## Device and compiled kernels\n\n```\n");
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g) {
        printf("```\nno usable OpenCL device\n");
        fprintf(stderr, "no usable OpenCL device; see %s\n", path);
        return 2;
    }
    fbm_gpu_info(g, stdout);
    printf("```\n\n## Correctness tests\n\n```\n");
    fflush(stdout);
    fprintf(stderr, "[1/5] correctness tests...\n");
    const int f = run_tests(g);
    printf("```\n\n## Issue-rate probe\n\n");
    fflush(stdout);
    fprintf(stderr, "[2/5] issue-rate probe...\n");
    fbm_gpu_probe(g, stdout);
    printf("\n## Benchmark: nonce vs vr layout\n\n");
    fflush(stdout);
    fprintf(stderr, "[3/5] interleaved benchmark...\n");
    signal(SIGINT, on_sigint);
    run_bench(g, "poclbm,nonce,vr", 10, 2, FBM_VR_MAX);
    printf("\nWith fewer versions per nonce (pools with a narrow BIP 310 mask):\n\n");
    run_bench(g, "poclbm,nonce,vr", 5, 1, 64);
    printf("\n## Sustained runs\n\n");
    fflush(stdout);
    if (sus > 0) {
        fprintf(stderr, "[4/5] sustained runs, %.0f s per layout (Ctrl-C skips)...\n", sus);
        run_sustain(g, FBM_GPU_VR, sus, FBM_VR_MAX);
        stop_requested = 0;
        run_sustain(g, FBM_GPU_NONCE, sus, 1);
    }
    printf("\n## Driver-compiled binary\n\n");
    fprintf(stderr, "[5/5] saving the driver-compiled binary...\n");
    fbm_gpu_dump(g, "fbm-gpu-kernels.bin");
    printf("Audit it with `python3 tools/gpu_isacheck.py --binary fbm-gpu-kernels.bin` (needs\n"
           "llvm-objdump), or send it back with this report.\n\n"
           "## Power tuning (manual)\n\n"
           "The power limit moves efficiency more than any kernel change. To measure J/TH at\n"
           "several limits: set the limit (Adrenalin: Performance > Tuning > Power Limit;\n"
           "Linux: echo WATTS000000 > /sys/class/drm/cardN/device/hwmon/hwmonM/power1_cap),\n"
           "then run `fbm-gpu sustain --seconds 300` and note the board power from the\n"
           "overlay (Windows) or from the output (Linux). If the tests or the mining canary\n"
           "ever fail after an undervolt, the card is not stable at that setting.\n");
    fbm_gpu_close(g);
    fflush(stdout);
    fprintf(stderr, "done: %s%s\n", path, f ? " (TESTS FAILED: see the report)" : "");
    return f ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage();
        return 2;
    }
    const char *cmd = argv[1];
    argc--;
    argv++;
    if (strcmp(cmd, "list") == 0)
        return fbm_gpu_list() ? 0 : 1;
    if (strcmp(cmd, "test") == 0)
        return cmd_test(argc, argv);
    if (strcmp(cmd, "bench") == 0)
        return cmd_bench(argc, argv);
    if (strcmp(cmd, "mine") == 0)
        return cmd_mine(argc, argv);
    if (strcmp(cmd, "probe") == 0)
        return cmd_probe(argc, argv);
    if (strcmp(cmd, "sustain") == 0)
        return cmd_sustain(argc, argv);
    if (strcmp(cmd, "report") == 0)
        return cmd_report(argc, argv);
    if (strcmp(cmd, "info") == 0 || strcmp(cmd, "dump") == 0) {
        fbm_gpu *g = open_from_args(argc, argv);
        int rc = 0;
        if (!g)
            return 2;
        if (strcmp(cmd, "info") == 0)
            fbm_gpu_info(g, stdout);
        else
            rc = fbm_gpu_dump(g, arg_value(argc, argv, "--out", "fbm-gpu-kernels.bin")) ? 1 : 0;
        fbm_gpu_close(g);
        return rc;
    }
    usage();
    return 2;
}
