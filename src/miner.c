#define _GNU_SOURCE
#include "miner.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

typedef struct {
    const fbm_kernel *k;
    const fbm_job *job;
    uint32_t r0, nr, n0;
    uint64_t nn;
    int cpu;
    fbm_hits hits;
    uint64_t hashes;
} worker;

static void *worker_main(void *arg)
{
    worker *w = arg;
    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(w->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    w->hashes = w->nn ? w->k->scan(w->job, w->r0, w->nr, w->n0, w->nn, &w->hits) : 0;
    return NULL;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

fbm_run_stats fbm_run(const fbm_kernel *k, const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                      uint64_t nn, int threads, fbm_hits *out)
{
    fbm_run_stats st = {0, 0, 0};
    worker *ws = calloc((size_t)threads, sizeof *ws);
    pthread_t *tids = calloc((size_t)threads, sizeof *tids);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    /* Slices are multiples of 64 nonces so SIMD lanes stay full. */
    uint64_t slice = (nn + (uint64_t)threads - 1) / (uint64_t)threads;
    slice = (slice + 63) & ~(uint64_t)63;

    for (int t = 0; t < threads; t++) {
        uint64_t start = (uint64_t)t * slice;
        ws[t].k = k;
        ws[t].job = job;
        ws[t].r0 = r0;
        ws[t].nr = nr;
        ws[t].n0 = (uint32_t)(n0 + start);
        ws[t].nn = start >= nn ? 0 : (nn - start < slice ? nn - start : slice);
        ws[t].cpu = threads <= ncpu ? t : -1;
        /* Each worker can hold everything the caller can. */
        ws[t].hits.cap = out->cap;
        ws[t].hits.v = calloc(out->cap ? out->cap : 1, sizeof(fbm_hit));
    }

    double t0 = now();
    uint64_t c0 = __rdtsc();
    for (int t = 0; t < threads; t++)
        pthread_create(&tids[t], NULL, worker_main, &ws[t]);
    for (int t = 0; t < threads; t++)
        pthread_join(tids[t], NULL);
    st.tsc = __rdtsc() - c0;
    st.seconds = now() - t0;

    for (int t = 0; t < threads; t++) {
        st.hashes += ws[t].hashes;
        size_t stored = ws[t].hits.n < ws[t].hits.cap ? ws[t].hits.n : ws[t].hits.cap;
        for (size_t i = 0; i < stored; i++)
            fbm_hits_push(out, ws[t].hits.v[i].version, ws[t].hits.v[i].nonce);
        /* Count candidates a worker could not store, so the caller sees
         * out->n > out->cap instead of a silently short list. */
        out->n += ws[t].hits.n - stored;
        free(ws[t].hits.v);
    }
    free(ws);
    free(tids);
    return st;
}
