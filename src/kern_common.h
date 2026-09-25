/* Helpers shared by the generated-kernel wrappers. */
#ifndef FBM_KERN_COMMON_H
#define FBM_KERN_COMMON_H

#include <stdint.h>
#include <string.h>

#include "header.h"
#include "kernels.h"
#include "sha256.h"
#include "util.h"

/* Scalar primitives used by the generated precompute functions. */
#define S_ROTR(x, n) fbm_rotr32((x), (n))
#define S_BSIG0(x) (S_ROTR(x, 2) ^ S_ROTR(x, 13) ^ S_ROTR(x, 22))
#define S_BSIG1(x) (S_ROTR(x, 6) ^ S_ROTR(x, 11) ^ S_ROTR(x, 25))
#define S_SSIG0(x) (S_ROTR(x, 7) ^ S_ROTR(x, 18) ^ ((x) >> 3))
#define S_SSIG1(x) (S_ROTR(x, 17) ^ S_ROTR(x, 19) ^ ((x) >> 10))
#define S_CH(e, f, g) ((g) ^ ((e) & ((f) ^ (g))))
#define S_MAJ(a, b, c) (((a) & (b)) | ((c) & ((a) | (b))))

/* Generated-kernel inputs for one version of a job:
 * in[0..7] = midstate (state after the first 64 header bytes),
 * in[8..10] = block-2 message words W0..W2 (merkle tail, time, bits). */
static inline void fbm_job_inputs(const uint8_t header[FBM_HEADER_LEN], uint32_t version,
                                  uint32_t in[11])
{
    uint8_t block[64];
    uint32_t st[8];

    memcpy(block, header, sizeof block);
    fbm_store_le32(block + FBM_OFF_VERSION, version);
    memcpy(st, fbm_sha256_iv, sizeof st);
    fbm_sha256_transform(st, block);
    memcpy(in, st, sizeof st);
    in[8] = fbm_load_be32(header + 64);
    in[9] = fbm_load_be32(header + FBM_OFF_TIME);
    in[10] = fbm_load_be32(header + FBM_OFF_BITS);
}

#endif
