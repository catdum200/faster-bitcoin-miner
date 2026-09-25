/* Benchmark pooler/cpuminer's scanhash_sha256d (hand-written x86-64 assembly:
 * midstate, pre-extension and early exit, 8-way AVX2 when available) on the
 * same machine as fbm, with the same timing method: fixed nonce count per
 * thread, one thread per core, median of repeats. Build: ./build.sh */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "miner.h"

struct work_restart *work_restart;

/* Only reached for candidates (1 in 2^32 per hash with this target). */
bool fulltest(const uint32_t *hash, const uint32_t *target)
{
    for (int i = 7; i >= 0; i--) {
        if (hash[i] != target[i])
            return hash[i] < target[i];
    }
    return true;
}

typedef struct {
    int id;
    uint32_t count;
    unsigned long done;
} task;

static void *run(void *arg)
{
    task *t = arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(t->id, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);

    uint32_t pdata[32], target[8];
    for (int i = 0; i < 32; i++)
        pdata[i] = 0x9e3779b9u * (i + 1) ^ 0x12345678u;
    memset(target, 0, sizeof target); /* hard target: no early returns */
    pdata[19] = (uint32_t)t->id << 28;
    uint32_t first = pdata[19];
    unsigned long done = 0, total = 0;
    while (total < t->count) {
        scanhash_sha256d(t->id, pdata, target, first + t->count - 1, &done);
        total += done;
        pdata[19]++;
    }
    t->done = total;
    return NULL;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    int threads = argc > 1 ? atoi(argv[1]) : 1;
    uint32_t count = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : 1u << 24;
    int repeats = argc > 3 ? atoi(argv[3]) : 5;
    work_restart = calloc(threads, sizeof *work_restart);
    printf("cpuminer sha256d: 4-way=%d 8-way=%d\n", sha256_use_4way(), sha256_use_8way());
    double rates[64];
    for (int r = 0; r < repeats; r++) {
        pthread_t tid[64];
        task ts[64];
        double t0 = now();
        for (int i = 0; i < threads; i++) {
            ts[i] = (task){i, count, 0};
            pthread_create(&tid[i], NULL, run, &ts[i]);
        }
        unsigned long total = 0;
        for (int i = 0; i < threads; i++) {
            pthread_join(tid[i], NULL);
            total += ts[i].done;
        }
        rates[r] = total / (now() - t0) / 1e6;
    }
    qsort(rates, repeats, sizeof(double), cmp);
    printf("cpuminer     %7d %12.3f %10.3f..%.3f MH/s (median, min..max)\n", threads,
           rates[repeats / 2], rates[0], rates[repeats - 1]);
    return 0;
}
