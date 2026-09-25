/* fbm-gpu: the OpenCL (GPU) front end. It has no CPU-specific code, so it
 * builds wherever there is a C11 compiler and an OpenCL driver, Windows
 * included. */
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

static const char *const layout_name[] = {"nonce", "vr"};

static void usage(void)
{
    fprintf(stderr,
            "usage:\n"
            "  fbm-gpu list                      OpenCL platforms and devices\n"
            "  fbm-gpu info                      device and compiled-kernel resources\n"
            "  fbm-gpu test                      correctness tests against the CPU reference\n"
            "  fbm-gpu bench [options]           interleaved hash-rate benchmark\n"
            "      --kernels nonce,vr (default both)  --rounds R (default 10)\n"
            "      --seconds S per run (default 2)     --versions N for vr (default 65536)\n"
            "  fbm-gpu mine --header HEX80 [options] search for valid (version, nonce) pairs\n"
            "      --start NONCE --count N (default: whole 2^32 range)\n"
            "      --versions N (BIP 320 rolled versions, default 1; 64+ uses vr)\n"
            "      --kernel nonce|vr\n"
            "  fbm-gpu dump --out FILE           save the driver-compiled kernels (AMD: ELF,\n"
            "                                    for tools/gpu_isacheck.py)\n"
            "common options:\n"
            "  --platform P --device D           pick a device (default: first GPU)\n"
            "  --wg N                            work-group size (default 64)\n"
            "  --nonce-iters N --vr-iters N      nonces per work-item per launch (16, 128)\n"
            "  --launch-ms MS                    target kernel launch length (default 100)\n");
}

static const char *arg_value(int argc, char **argv, const char *name, const char *def)
{
    for (int i = 0; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    }
    return def;
}

static double now(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Launch sizes are set in hashes; --launch-ms converts at a nominal 3 GH/s
 * (an RX 9060 XT-class estimate), so slower devices get longer launches. */
static fbm_gpu *open_from_args(int argc, char **argv)
{
    fbm_gpu_opts o;
    fbm_gpu_default_opts(&o);
    o.platform = atoi(arg_value(argc, argv, "--platform", "-1"));
    o.device = atoi(arg_value(argc, argv, "--device", "-1"));
    o.wg = (unsigned)atoi(arg_value(argc, argv, "--wg", "64"));
    o.nonce_iters = (unsigned)atoi(arg_value(argc, argv, "--nonce-iters", "16"));
    o.vr_iters = (unsigned)atoi(arg_value(argc, argv, "--vr-iters", "128"));
    o.launch_hashes = atof(arg_value(argc, argv, "--launch-ms", "100")) * 3e6;
    if (o.wg == 0 || o.wg > 1024 || o.nonce_iters == 0 || o.vr_iters == 0 ||
        o.launch_hashes < 1) {
        fprintf(stderr, "bad tuning option\n");
        return NULL;
    }
    fbm_gpu *g = fbm_gpu_open(&o);
    if (g)
        printf("device: %s\n", fbm_gpu_name(g));
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
    printf("  %-34s %zu candidates\n", label, want.n);
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

static int cmd_test(int argc, char **argv)
{
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g)
        return 2;
    fbm_gpu_opts *o = fbm_gpu_options(g);
    failures = 0;
    test_differential(g, "default launch sizes");
    /* Tiny launches: many launches per scan, partial work-groups, iteration
     * counts that do not divide the range. */
    const fbm_gpu_opts saved = *o;
    o->launch_hashes = 1000;
    o->nonce_iters = 3;
    o->vr_iters = 7;
    test_differential(g, "tiny launches (1000 hashes, 3 and 7 nonces per work-item)");
    *o = saved;
    test_known(g);
    test_overflow(g);
    printf(failures ? "%d FAILURE(S)\n" : "all GPU tests passed\n", failures);
    fbm_gpu_close(g);
    return failures ? 1 : 0;
}

/* ---- bench --------------------------------------------------------------- */

#define MAX_ROUNDS 200

typedef struct {
    fbm_gpu_layout layout;
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
    uint64_t n = fbm_gpu_scan(g, e->layout, job, 0, e->nr, 0, e->nn, &hits);
    const double dt = now() - t0;
    *krate = n / fbm_gpu_kernel_seconds(g) / 1e6;
    return n / dt / 1e6;
}

static int cmd_bench(int argc, char **argv)
{
    const char *which = arg_value(argc, argv, "--kernels", "nonce,vr");
    int rounds = atoi(arg_value(argc, argv, "--rounds", "10"));
    double seconds = atof(arg_value(argc, argv, "--seconds", "2"));
    uint32_t versions = (uint32_t)strtoul(arg_value(argc, argv, "--versions", "65536"), NULL, 0);
    bench_entry es[2];
    int ne = 0;
    fbm_job job;

    if (rounds < 1 || rounds > MAX_ROUNDS || seconds <= 0 || versions == 0 ||
        versions > FBM_VR_MAX) {
        usage();
        return 2;
    }
    if (strstr(which, "nonce"))
        es[ne++] = (bench_entry){.layout = FBM_GPU_NONCE, .nr = 1};
    if (strstr(which, "vr"))
        es[ne++] = (bench_entry){.layout = FBM_GPU_VR, .nr = versions};
    if (!ne) {
        usage();
        return 2;
    }
    fbm_gpu *g = open_from_args(argc, argv);
    if (!g)
        return 2;

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

    printf("| kernel | MH/s median (wall) | IQR | min..max | MH/s (kernel time) |%s\n",
           ne == 2 ? " vr / nonce [95% CI] |" : "");
    printf("|---|---:|---:|---:|---:|%s\n", ne == 2 ? "---:|" : "");
    for (int i = 0; i < ne; i++) {
        double s[MAX_ROUNDS];
        memcpy(s, es[i].rate, sizeof(double) * rounds);
        qsort(s, rounds, sizeof(double), fbm_cmp_double);
        printf("| %s | %.1f | %.1f..%.1f | %.1f..%.1f | %.1f |", layout_name[es[i].layout],
               fbm_quantile(s, rounds, 0.5), fbm_quantile(s, rounds, 0.25),
               fbm_quantile(s, rounds, 0.75), s[0], s[rounds - 1],
               fbm_median(es[i].krate, rounds));
        if (ne == 2 && es[i].layout == FBM_GPU_VR) {
            double ratio[MAX_ROUNDS], lo, hi;
            for (int r = 0; r < rounds; r++)
                ratio[r] = es[i].rate[r] / es[1 - i].rate[r];
            fbm_bootstrap_ci(ratio, rounds, &lo, &hi);
            printf(" %.3fx [%.3f, %.3f] |", fbm_median(ratio, rounds), lo, hi);
        } else if (ne == 2) {
            printf(" |");
        }
        printf("\n");
    }
    printf("\nwall = scan time on the host clock, including launch gaps and the schedule\n"
           "pre-pass; kernel time = sum of OpenCL profiling intervals.\n");
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

    /* Scan in slices of ~2^32 hashes so progress and solutions show up
     * while it runs (~1-2 s each on a fast GPU). */
    const uint64_t slice = versions >= (1u << 16) ? 1 << 16 : (1ull << 32) / versions;
    int solutions = 0;
    size_t candidates = 0;
    uint64_t hashes = 0;
    const double t0 = now();
    for (uint64_t off = 0; off < count; off += slice) {
        const uint64_t nn = count - off < slice ? count - off : slice;
        fbm_hit buf[4096];
        fbm_hits hits = {buf, 4096, 0};
        hashes += fbm_gpu_scan(g, layout, &job, 0, versions, (uint32_t)(start + off), nn, &hits);
        candidates += hits.n;
        for (size_t i = 0; i < hits.n && i < hits.cap; i++) {
            uint8_t hdr[80], hash[32];
            char disp[65], full[161];
            memcpy(hdr, job.header, 80);
            fbm_header_set(hdr, FBM_OFF_VERSION, hits.v[i].version);
            fbm_header_set(hdr, FBM_OFF_NONCE, hits.v[i].nonce);
            fbm_sha256d(hdr, 80, hash);
            if (fbm_top32_le(hash) > job.t7) {
                /* The filter is exact, so this can only be a kernel or driver
                 * bug, which could also be dropping real blocks. */
                fprintf(stderr, "BUG: GPU reported version 0x%08x nonce %u, which fails its own "
                        "pre-filter; aborting\n", hits.v[i].version, hits.v[i].nonce);
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
        const double dt = now() - t0;
        fprintf(stderr, "\r%.1f%% %.1f MH/s", 100.0 * (off + nn) / count, hashes / dt / 1e6);
    }
    const double dt = now() - t0;
    fprintf(stderr, "\n");
    printf("%llu hashes in %.3f s = %.3f MH/s, %d solution(s), %zu candidate(s)\n",
           (unsigned long long)hashes, dt, hashes / dt / 1e6, solutions, candidates);
    fbm_gpu_close(g);
    return solutions ? 0 : 1;
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
