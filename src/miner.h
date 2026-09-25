/* Multithreaded driver: splits a search rectangle across threads. */
#ifndef FBM_MINER_H
#define FBM_MINER_H

#include <stdint.h>

#include "kernels.h"

typedef struct {
    uint64_t hashes;
    double seconds;
    uint64_t tsc; /* TSC ticks elapsed (wall clock, not per thread) */
} fbm_run_stats;

/* Hash (r0..r0+nr) x (n0..n0+nn) with `threads` threads, each owning a
 * contiguous slice of the nonce range. Candidates from all threads are
 * appended to *out. Threads are pinned to CPUs 0..threads-1 when possible. */
fbm_run_stats fbm_run(const fbm_kernel *k, const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                      uint64_t nn, int threads, fbm_hits *out);

#endif
