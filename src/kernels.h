/* Common interface for every hashing kernel.
 *
 * A kernel hashes every (version, nonce) pair of a rectangle of the search
 * space and reports "candidates": pairs whose final hash has
 * top32(hash as LE 256-bit number) <= job->t7. That is a necessary condition
 * for hash <= target, so the caller confirms each candidate with the
 * reference hash. With t7 = 0 (any real network target) the test is
 * "last 32 bits of the hash are zero", which lets fast kernels stop
 * after round 60 of the second SHA-256.
 */
#ifndef FBM_KERNELS_H
#define FBM_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* BIP 320: version bits 13..28 are free for miners to roll. We roll by XOR,
 * so r = 0 is exactly the job's own version. */
#define FBM_VR_SHIFT 13
#define FBM_VR_MAX (1u << 16)

typedef struct {
    uint8_t header[80]; /* nonce field ignored; version is the rolling base */
    uint32_t t7;        /* candidate iff top32(hash) <= t7 */
} fbm_job;

typedef struct {
    uint32_t version;
    uint32_t nonce;
} fbm_hit;

typedef struct {
    fbm_hit *v;
    size_t cap;
    size_t n; /* candidates found; only the first cap are stored */
} fbm_hits;

static inline void fbm_hits_push(fbm_hits *h, uint32_t version, uint32_t nonce)
{
    if (h->n < h->cap) {
        h->v[h->n].version = version;
        h->v[h->n].nonce = nonce;
    }
    h->n++;
}

static inline uint32_t fbm_rolled_version(uint32_t base, uint32_t r)
{
    return base ^ (r << FBM_VR_SHIFT);
}

/* Hash every pair (version r, nonce n) with r0 <= r < r0 + nr and
 * n0 <= n < n0 + nn. Requires r0 + nr <= FBM_VR_MAX and
 * n0 + nn <= 2^32. Returns the number of hashes attempted (nr * nn). */
typedef uint64_t (*fbm_scan_fn)(const fbm_job *job, uint32_t r0, uint32_t nr,
                                uint32_t n0, uint64_t nn, fbm_hits *out);

typedef struct {
    const char *name;
    const char *desc;
    fbm_scan_fn scan;
    int (*supported)(void);
    /* Kernels that put versions in SIMD lanes want nr to be a multiple of
     * this; others use 1. */
    uint32_t version_lanes;
    /* 1 if the kernel shares one block-2 schedule across versions. */
    int vr;
} fbm_kernel;

size_t fbm_kernel_count(void);
const fbm_kernel *fbm_kernel_at(size_t i);
const fbm_kernel *fbm_kernel_find(const char *name);
/* Fastest kernel supported by the running CPU. */
const fbm_kernel *fbm_kernel_best(void);

#endif
