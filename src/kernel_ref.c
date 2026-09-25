/* T0 baselines: what a straightforward miner does. The full 80-byte header
 * is hashed from scratch for every nonce (3 SHA-256 compressions plus
 * padding and buffering), first with our portable reference SHA-256 and then
 * with OpenSSL's SHA256(). */
#include <openssl/sha.h>
#include <string.h>

#include "header.h"
#include "kernels.h"
#include "sha256.h"

uint64_t fbm_scan_ref(const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                      uint64_t nn, fbm_hits *out)
{
    uint8_t hdr[FBM_HEADER_LEN], hash[32];
    uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);

    memcpy(hdr, job->header, sizeof hdr);
    for (uint32_t r = r0; r < r0 + nr; r++) {
        uint32_t version = fbm_rolled_version(base, r);
        fbm_header_set(hdr, FBM_OFF_VERSION, version);
        for (uint64_t i = 0; i < nn; i++) {
            uint32_t nonce = (uint32_t)(n0 + i);
            fbm_header_set(hdr, FBM_OFF_NONCE, nonce);
            fbm_sha256d(hdr, sizeof hdr, hash);
            if (fbm_top32_le(hash) <= job->t7)
                fbm_hits_push(out, version, nonce);
        }
    }
    return (uint64_t)nr * nn;
}

uint64_t fbm_scan_openssl(const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                          uint64_t nn, fbm_hits *out)
{
    uint8_t hdr[FBM_HEADER_LEN], first[32], hash[32];
    uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);

    memcpy(hdr, job->header, sizeof hdr);
    for (uint32_t r = r0; r < r0 + nr; r++) {
        uint32_t version = fbm_rolled_version(base, r);
        fbm_header_set(hdr, FBM_OFF_VERSION, version);
        for (uint64_t i = 0; i < nn; i++) {
            uint32_t nonce = (uint32_t)(n0 + i);
            fbm_header_set(hdr, FBM_OFF_NONCE, nonce);
            SHA256(hdr, sizeof hdr, first);
            SHA256(first, sizeof first, hash);
            if (fbm_top32_le(hash) <= job->t7)
                fbm_hits_push(out, version, nonce);
        }
    }
    return (uint64_t)nr * nn;
}
