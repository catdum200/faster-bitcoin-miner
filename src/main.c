/* fbm: faster bitcoin miner - command line front end. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "header.h"
#include "kernels.h"
#include "freq.h"
#include "miner.h"
#include "selftest.h"
#include "sha256.h"
#include "stats.h"

static void usage(void)
{
    fprintf(stderr,
            "usage:\n"
            "  fbm test                         run correctness tests\n"
            "  fbm list                         list kernels and CPU support\n"
            "  fbm freq                         measure core clock for scalar/ymm/zmm code\n"
            "  fbm bench [options]              interleaved hash-rate benchmark\n"
            "      --kernels a,b,c|all (default all)  --threads N (default 1)\n"
            "      --seconds S per run (default 1)    --rounds R (default 20)\n"
            "      --baseline NAME (default cpuminer-opt16 if built, else ref)\n"
            "  fbm mine --header HEX80 [options] search for valid (version, nonce) pairs\n"
            "      --kernel NAME (default: fastest for the version count)\n"
            "      --threads N (default: all CPUs)\n"
            "      --start NONCE --count N (default: whole 2^32 range)\n"
            "      --versions N (BIP 320 rolled versions, default 1; 64+ selects\n"
            "                    the version-rolling kernels)\n");
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
    printf("fastest supported: %s (1 version per nonce), %s (64+ rolled versions)\n",
           fbm_kernel_best(1)->name, fbm_kernel_best(FBM_VR_MAX)->name);
    return 0;
}

static int cmd_freq(void)
{
    printf("core clock (GHz), dependent-add chains:\n");
    printf("  scalar           %.2f\n", fbm_ghz_scalar());
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2"))
        printf("  256-bit (ymm)    %.2f\n", fbm_ghz_ymm());
    if (__builtin_cpu_supports("avx512f"))
        printf("  512-bit (zmm)    %.2f\n", fbm_ghz_zmm());
    return 0;
}

/* Versions scanned per run: version-rolling kernels need several versions
 * per nonce to share the message schedule. */
static uint32_t bench_versions(const fbm_kernel *k)
{
    return k->vr ? 128 : 1;
}

#define MAX_ROUNDS 200

typedef struct {
    const fbm_kernel *k;
    char label[48];
    uint32_t nr;
    uint64_t nn; /* nonces per thread per run */
    double rate[MAX_ROUNDS]; /* MH/s, all threads */
} bench_entry;

static double bench_once(bench_entry *e, const fbm_job *job, int threads)
{
    fbm_hit buf[64];
    fbm_hits hits = {buf, 64, 0};
    fbm_run_stats st = fbm_run(e->k, job, 0, e->nr, 0, e->nn * threads, threads, &hits);
    return st.hashes / st.seconds / 1e6;
}

/* Interleaved benchmark: every round runs each kernel once (in rotating
 * order) so slow drifts and noisy neighbours hit all kernels alike. Speedups
 * are medians of per-round ratios with bootstrap 95% CIs. */
static int cmd_bench(int argc, char **argv)
{
    const char *which = arg_value(argc, argv, "--kernels", arg_value(argc, argv, "--kernel", "all"));
    int threads = atoi(arg_value(argc, argv, "--threads", "1"));
    double seconds = atof(arg_value(argc, argv, "--seconds", "1"));
    int rounds = atoi(arg_value(argc, argv, "--rounds", "20"));
    const char *base_name = arg_value(argc, argv, "--baseline", NULL);
    static bench_entry es[32];
    int ne = 0;
    fbm_job job;

    if (threads < 1 || rounds < 1 || rounds > MAX_ROUNDS || seconds <= 0) {
        usage();
        return 2;
    }
    /* The list is either "all" or comma-separated names; "name@N" runs a
     * kernel with N rolled versions per nonce (default 128 for
     * version-rolling kernels, else 1). */
    char list[1024];
    snprintf(list, sizeof list, "%s", which);
    for (char *tok = strtok(list, ","); tok && ne < 32; tok = strtok(NULL, ",")) {
        char *at = strchr(tok, '@');
        uint32_t nr = 0;
        if (at) {
            *at = '\0';
            nr = (uint32_t)strtoul(at + 1, NULL, 0);
        }
        for (size_t i = 0; i < fbm_kernel_count() && ne < 32; i++) {
            const fbm_kernel *k = fbm_kernel_at(i);
            if (strcmp(tok, "all") != 0 && strcmp(tok, k->name) != 0)
                continue;
            if (!k->supported()) {
                printf("skipping %s: not supported on this CPU\n", k->name);
                continue;
            }
            es[ne].k = k;
            es[ne].nr = nr ? nr : bench_versions(k);
            if (es[ne].nr > FBM_VR_MAX) {
                fprintf(stderr, "too many versions\n");
                return 2;
            }
            snprintf(es[ne].label, sizeof es[ne].label, "%s%s%s", k->name, at ? "@" : "",
                     at ? at + 1 : "");
            ne++;
        }
    }
    if (ne == 0) {
        fprintf(stderr, "no kernels selected\n");
        return 2;
    }
    int base = -1;
    for (int i = 0; i < ne; i++) {
        if (base_name ? !strcmp(es[i].label, base_name)
                      : (!strcmp(es[i].k->name, "cpuminer-opt16") ||
                         (base < 0 && !strcmp(es[i].k->name, "ref"))))
            base = i;
    }

    /* A fixed pseudo-random header; t7 = 0 as for any real network target. */
    for (int i = 0; i < 80; i++)
        job.header[i] = (uint8_t)(i * 37 + 11);
    job.t7 = 0;

    double ghz[3] = {fbm_ghz_scalar(), 0, 0};
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2"))
        ghz[1] = fbm_ghz_ymm();
    if (__builtin_cpu_supports("avx512f"))
        ghz[2] = fbm_ghz_zmm();
    printf("threads %d, %d interleaved rounds of ~%.2f s per kernel; core GHz: scalar %.2f, "
           "ymm %.2f, zmm %.2f\n",
           threads, rounds, seconds, ghz[0], ghz[1], ghz[2]);

    /* Calibrate each kernel's run length, which doubles as a warm-up. */
    for (int i = 0; i < ne; i++) {
        es[i].nn = 256;
        for (;;) {
            fbm_hit buf[64];
            fbm_hits hits = {buf, 64, 0};
            fbm_run_stats st = fbm_run(es[i].k, &job, 0, es[i].nr, 0, es[i].nn * threads, threads,
                                       &hits);
            if (st.seconds > 0.1 || es[i].nn >= (1ull << 31) / threads) {
                double per = st.seconds / (double)es[i].nn;
                es[i].nn = (uint64_t)(seconds / per);
                break;
            }
            es[i].nn *= 4;
        }
        if (es[i].nn * threads > 0xffffffffull)
            es[i].nn = 0xffffffffull / threads;
        es[i].nn = es[i].nn < 64 ? 64 : es[i].nn;
    }

    for (int r = 0; r < rounds; r++) {
        for (int j = 0; j < ne; j++) {
            int i = (j + r) % ne;
            es[i].rate[r] = bench_once(&es[i], &job, threads);
        }
        fprintf(stderr, "\rround %d/%d", r + 1, rounds);
    }
    fprintf(stderr, "\r                \r");

    printf("| kernel | MH/s median | IQR | min..max | ns/hash/core | ~core cycles/hash |");
    if (base >= 0)
        printf(" vs %s [95%% CI] |", es[base].label);
    printf("\n|---|---:|---:|---:|---:|---:|%s\n", base >= 0 ? "---:|" : "");
    for (int i = 0; i < ne; i++) {
        double sorted[MAX_ROUNDS];
        memcpy(sorted, es[i].rate, sizeof(double) * rounds);
        qsort(sorted, rounds, sizeof(double), fbm_cmp_double);
        double med = fbm_quantile(sorted, rounds, 0.5);
        double ns = threads * 1e3 / med; /* ns per hash per core */
        int low = 0;
        for (int r = 0; r < rounds; r++)
            low += es[i].rate[r] < 0.8 * med;
        printf("| %s | %.3f | %.3f..%.3f | %.3f..%.3f | %.2f | %.0f |", es[i].label, med,
               fbm_quantile(sorted, rounds, 0.25), fbm_quantile(sorted, rounds, 0.75), sorted[0],
               sorted[rounds - 1], ns, ns * ghz[es[i].k->isa]);
        if (base >= 0) {
            double ratio[MAX_ROUNDS], lo, hi;
            for (int r = 0; r < rounds; r++)
                ratio[r] = es[i].rate[r] / es[base].rate[r];
            fbm_bootstrap_ci(ratio, rounds, &lo, &hi);
            printf(" %.3fx [%.3f, %.3f] |", fbm_median(ratio, rounds), lo, hi);
        }
        if (low)
            printf(" (%d run(s) < 80%% of median)", low);
        printf("\n");
    }
    printf("\nns/hash/core = threads / rate. ~core cycles/hash = ns/hash/core x measured core GHz\n"
           "for the kernel's widest vectors (scalar, ymm or zmm license).\n");
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
    const fbm_kernel *k = kname ? fbm_kernel_find(kname) : fbm_kernel_best(versions);
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
        if (fbm_top32_le(hash) > job.t7) {
            /* The kernel's filter is exact, so this can only be a kernel bug
             * (e.g. byte order), which could also be dropping real blocks. */
            fprintf(stderr, "BUG: kernel %s reported version 0x%08x nonce %u, which fails its own "
                    "pre-filter; aborting\n", k->name, hits.v[i].version, hits.v[i].nonce);
            return 3;
        }
        if (fbm_cmp256_le(hash, target) > 0)
            continue; /* top 32 bits pass, full 256-bit target does not */
        solutions++;
        fbm_hash_display(hash, disp);
        fbm_hex_encode(hdr, 80, full);
        printf("SOLUTION version=0x%08x nonce=%u hash=%s\n  header=%s\n", hits.v[i].version,
               hits.v[i].nonce, disp, full);
    }
    if (hits.n > hits.cap)
        fprintf(stderr, "warning: %zu candidates found but only %zu checked (buffer full); "
                "narrow the range\n", hits.n, hits.cap);
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
    if (strcmp(argv[1], "freq") == 0)
        return cmd_freq();
    if (strcmp(argv[1], "bench") == 0)
        return cmd_bench(argc - 1, argv + 1);
    if (strcmp(argv[1], "mine") == 0)
        return cmd_mine(argc - 1, argv + 1);
    usage();
    return 2;
}
