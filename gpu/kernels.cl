/* OpenCL kernels for fbm-gpu (see gpu/prelude.cl for how the program is
 * assembled). The SHA-256d arithmetic is all generated; this file only
 * decides which values live where:
 *
 *   JOB values    job[]    per job, identical in every lane: kernel buffer,
 *                          uniform loads (scalar loads into SGPRs)
 *   LANE values   lanes[]  per rolled version: loaded once per work-item into
 *                          VGPRs, reused for every nonce it hashes
 *   NONCE values  sched[]  per nonce, identical in every lane: computed by the
 *                          fbm_vr_sched pre-pass, read with uniform (scalar)
 *                          loads, so every lane of a wave shares one copy
 *
 * Buffers:
 *   job[0..63]   generated J[] slots; job[FBM_JOB_T7] = t7 (candidate iff
 *                bswap(H7) <= t7, the same compare as the CPU kernels)
 *   hits[0]      candidates found (may exceed the capacity: never dropped
 *                silently, the host reports overflow)
 *   hits[1]      capacity in pairs; hits[2 + 2i], hits[3 + 2i] = (index, nonce)
 *
 * FBM_WG (work-group size) is set by the host at build time. */

#define FBM_JOB_T7 64
/* NONCE row stride in words: 47 values padded to 48 = 3 x 64-byte loads. */
#define FBM_NNP ((G_RDNA_VR_NN + 15) / 16 * 16)

static inline void fbm_push(volatile __global uint *hits, uint index, uint nonce)
{
    const uint slot = atomic_inc(&hits[0]);
    if (slot < hits[1]) {
        hits[2 + 2 * slot] = index;
        hits[3 + 2 * slot] = nonce;
    }
}

/* Nonce-lane layout (prior art): every lane hashes the same version with its
 * own nonce. A work-group covers FBM_WG * iters consecutive nonces from
 * n0 + group * FBM_WG * iters; lanes at or past `count` compute but never
 * report. `r` is the version index, echoed in every hit. */
__kernel __attribute__((reqd_work_group_size(FBM_WG, 1, 1)))
void fbm_nonce(const __global uint *restrict job, uint n0, uint count, uint iters, uint r,
               volatile __global uint *restrict hits)
{
    uint J[G_RDNA_NONCE_NJ];
    for (int k = 0; k < G_RDNA_NONCE_NJ; k++)
        J[k] = job[k];
    const uint t7 = job[FBM_JOB_T7];
    const uint first = get_group_id(0) * (FBM_WG * iters) + get_local_id(0);

    for (uint i = 0; i < iters; i++) {
        const uint idx = first + i * FBM_WG;
        const uint nonce = n0 + idx;
        const uint h7 = g_rdna_nonce_body(J, 0, 0, BSWAP(nonce));
        if (BSWAP(h7) <= t7 && idx < count)
            fbm_push(hits, r, nonce);
    }
}

/* Schedule pre-pass for the version-lane layout: row j of sched[] holds the
 * generated NONCE values (block-2 message schedule plus round constants) of
 * nonce n0 + j. One work-item per nonce; the rows are then shared by every
 * version the main kernel hashes. */
__kernel __attribute__((reqd_work_group_size(FBM_WG, 1, 1)))
void fbm_vr_sched(const __global uint *restrict job, uint n0, uint count,
                  __global uint *restrict sched)
{
    uint J[G_RDNA_VR_NJ], N[G_RDNA_VR_NN];
    for (int k = 0; k < G_RDNA_VR_NJ; k++)
        J[k] = job[k];
    const uint j = get_group_id(0) * FBM_WG + get_local_id(0);
    if (j >= count)
        return;
    g_rdna_vr_nonce(BSWAP(n0 + j), J, N);
    for (int k = 0; k < G_RDNA_VR_NN; k++)
        sched[j * FBM_NNP + k] = N[k];
}

/* Version-lane layout (overt AsicBoost / BIP 320 on a GPU): lane v hashes
 * rolled version v; the whole work-group walks the same nonces, so the
 * schedule row is wave-uniform. Dimension 1 of the grid picks a block of
 * `iters` nonces. lanes[k * lane_stride + v] is LANE slot k of version v;
 * versions at or past `nr` are padding and never report. */
__kernel __attribute__((reqd_work_group_size(FBM_WG, 1, 1)))
void fbm_vr(const __global uint *restrict job, const __global uint *restrict lanes,
            uint lane_stride, uint nr, const __global uint *restrict sched, uint n0, uint count,
            uint iters, volatile __global uint *restrict hits)
{
    uint J[G_RDNA_VR_NJ], L[G_RDNA_VR_NL], N[G_RDNA_VR_NN];
    for (int k = 0; k < G_RDNA_VR_NJ; k++)
        J[k] = job[k];
    const uint t7 = job[FBM_JOB_T7];
    const uint v = get_group_id(0) * FBM_WG + get_local_id(0);
    for (int k = 0; k < G_RDNA_VR_NL; k++)
        L[k] = lanes[k * lane_stride + v];
    const uint first = get_group_id(1) * iters;

    for (uint i = 0; i < iters; i++) {
        const uint j = first + i; /* the same in every lane */
        if (j >= count)
            break;
        const __global uint *row = sched + j * FBM_NNP;
        for (int k = 0; k < G_RDNA_VR_NN; k++)
            N[k] = row[k];
        const uint h7 = g_rdna_vr_body(J, L, N, 0);
        if (BSWAP(h7) <= t7 && v < nr)
            fbm_push(hits, v, n0 + j);
    }
}
