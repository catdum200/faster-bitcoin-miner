/* fbm: faster bitcoin miner - command line front end. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "header.h"
#include "kernels.h"
#include "miner.h"
#include "selftest.h"
#include "sha256.h"

static void usage(void)
{
    fprintf(stderr,
            "usage:\n"
            "  fbm test                         run correctness tests\n"
            "  fbm list                         list kernels and CPU support\n"
            "  fbm bench [options]              measure hash rate of kernels\n"
            "      --kernel NAME|all  (default all)   --threads N (default 1)\n"
            "      --seconds S (per run, default 1)   --repeats R (default 5)\n"
            "  fbm mine --header HEX80 [options] search for valid (version, nonce) pairs\n"
            "      --kernel NAME (default: fastest)   --threads N (default: all CPUs)\n"
            "      --start NONCE --count N (default: whole 2^32 range)\n"
            "      --versions N (BIP 320 rolled versions, default 1)\n");
}

static const char *arg_value(int argc, char **argv, const char *name, const char *def)
{
    for (int i = 0; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    }
    return def;
}

static int cmd_list(void)
{
    for (size_t i = 0; i < fbm_kernel_count(); i++) {
        const fbm_kernel *k = fbm_kernel_at(i);
        printf("%-12s %-3s %s\n", k->name, k->supported() ? "yes" : "no", k->desc);
    }
    printf("fastest supported: %s\n", fbm_kernel_best()->name);
    return 0;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* Versions scanned per bench run: version-rolling kernels need several
 * versions per nonce to share the message schedule. */
static uint32_t bench_versions(const fbm_kernel *k)
{
    return k->vr ? 128 : 1;
}

static int cmd_bench(int argc, char **argv)
{
    const char *which = arg_value(argc, argv, "--kernel", "all");
    int threads = atoi(arg_value(argc, argv, "--threads", "1"));
    double seconds = atof(arg_value(argc, argv, "--seconds", "1"));
    int repeats = atoi(arg_value(argc, argv, "--repeats", "5"));
    fbm_job job;
    double ref_rate = 0;

    if (threads < 1 || repeats < 1 || repeats > 99 || seconds <= 0) {
        usage();
        return 2;
    }
    /* A fixed pseudo-random header; t7 = 0 as for any real network target. */
    for (int i = 0; i < 80; i++)
        job.header[i] = (uint8_t)(i * 37 + 11);
    job.t7 = 0;

    printf("%-12s %7s %12s %22s %14s %9s\n", "kernel", "threads", "MH/s(median)", "min..max",
           "cycles/hash*", "vs ref");
    for (size_t i = 0; i < fbm_kernel_count(); i++) {
        const fbm_kernel *k = fbm_kernel_at(i);
        if (strcmp(which, "all") != 0 && strcmp(which, k->name) != 0)
            continue;
        if (!k->supported()) {
            printf("%-12s (not supported on this CPU)\n", k->name);
            continue;
        }
        uint32_t nr = bench_versions(k);
        fbm_hit buf[256];
        fbm_hits hits = {buf, 256, 0};

        /* Calibrate the nonce count so one run takes about `seconds`. */
        uint64_t nn = 1024;
        fbm_run_stats st;
        for (;;) {
            st = fbm_run(k, &job, 0, nr, 0, nn * threads, threads, &hits);
            if (st.seconds > 0.05 || nn * nr * threads > (1ull << 40))
                break;
            nn *= 4;
        }
        double per_nonce = st.seconds / (double)(nn * threads);
        nn = (uint64_t)(seconds / per_nonce / threads);
        if (nn * threads > 0xffffffffull)
            nn = 0xffffffffull / threads;
        if (nn < 64)
            nn = 64;

        double rates[99], cycles[99];
        for (int r = 0; r < repeats; r++) {
            hits.n = 0;
            st = fbm_run(k, &job, 0, nr, 0, nn * threads, threads, &hits);
            rates[r] = st.hashes / st.seconds / 1e6;
            cycles[r] = (double)st.tsc * threads / st.hashes;
        }
        double sorted[99];
        memcpy(sorted, rates, sizeof(double) * repeats);
        qsort(sorted, repeats, sizeof(double), cmp_double);
        qsort(cycles, repeats, sizeof(double), cmp_double);
        double med = sorted[repeats / 2];
        if (strcmp(k->name, "ref") == 0)
            ref_rate = med;
        char range[64], rel[32] = "";
        snprintf(range, sizeof range, "%.3f..%.3f", sorted[0], sorted[repeats - 1]);
        if (ref_rate > 0)
            snprintf(rel, sizeof rel, "%.1fx", med / ref_rate);
        printf("%-12s %7d %12.3f %22s %14.0f %9s\n", k->name, threads, med, range,
               cycles[repeats / 2], rel);
        fflush(stdout);
    }
    printf("* TSC ticks (%s) x threads / hashes: cost of one hash on one core\n",
           "constant-rate timestamp counter");
    return 0;
}

static int cmd_mine(int argc, char **argv)
{
    const char *hex = arg_value(argc, argv, "--header", NULL);
    const char *kname = arg_value(argc, argv, "--kernel", NULL);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int threads = atoi(arg_value(argc, argv, "--threads", "0"));
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
    const fbm_kernel *k = kname ? fbm_kernel_find(kname) : fbm_kernel_best();
    if (!k || !k->supported()) {
        fprintf(stderr, "unknown or unsupported kernel\n");
        return 2;
    }
    if (threads <= 0)
        threads = (int)ncpu;
    job.t7 = fbm_top32_le(target);

    printf("kernel %s, %d threads, difficulty %.3f, %u version(s) x %llu nonces from %llu\n",
           k->name, threads, fbm_target_difficulty(target), versions, (unsigned long long)count,
           (unsigned long long)start);
    fbm_hit buf[4096];
    fbm_hits hits = {buf, 4096, 0};
    fbm_run_stats st = fbm_run(k, &job, 0, versions, (uint32_t)start, count, threads, &hits);

    int solutions = 0;
    for (size_t i = 0; i < hits.n && i < hits.cap; i++) {
        uint8_t hdr[80], hash[32];
        char disp[65], full[161];
        memcpy(hdr, job.header, 80);
        fbm_header_set(hdr, FBM_OFF_VERSION, hits.v[i].version);
        fbm_header_set(hdr, FBM_OFF_NONCE, hits.v[i].nonce);
        fbm_sha256d(hdr, 80, hash);
        if (fbm_cmp256_le(hash, target) > 0)
            continue; /* passed the 32-bit pre-filter but not the full target */
        solutions++;
        fbm_hash_display(hash, disp);
        fbm_hex_encode(hdr, 80, full);
        printf("SOLUTION version=0x%08x nonce=%u hash=%s\n  header=%s\n", hits.v[i].version,
               hits.v[i].nonce, disp, full);
    }
    printf("%llu hashes in %.3f s = %.3f MH/s, %d solution(s), %zu candidate(s)\n",
           (unsigned long long)st.hashes, st.seconds, st.hashes / st.seconds / 1e6, solutions,
           hits.n);
    return solutions ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage();
        return 2;
    }
    if (strcmp(argv[1], "test") == 0)
        return fbm_selftest();
    if (strcmp(argv[1], "list") == 0)
        return cmd_list();
    if (strcmp(argv[1], "bench") == 0)
        return cmd_bench(argc - 1, argv + 1);
    if (strcmp(argv[1], "mine") == 0)
        return cmd_mine(argc - 1, argv + 1);
    usage();
    return 2;
}
