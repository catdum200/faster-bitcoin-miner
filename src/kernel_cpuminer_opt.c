/* Optional baseline: cpuminer-opt's sha256d kernels (JayDDee/cpuminer-opt,
 * GPL-2), the fastest open-source CPU sha256d code we know of: midstate,
 * 3-round prehash, pre-extension, early exit, 16-way AVX-512. Built only by
 * `make BASELINES=1`, which fetches the source at a pinned release.
 *
 * The loop mirrors cpuminer-opt's scanhash_sha256d_16way/8way. The one
 * change: lanes step through nonces instead of big-endian W3 words (one byte
 * shuffle per vector), so fbm's differential tests can check this kernel
 * like any other. target[6] is all-ones so only the top word filters, which
 * is the same candidate rule as fbm's kernels. */
#include <immintrin.h>

#include "kern_common.h"

void sha256_16x32_prehash_3rounds(__m512i *, __m512i *, const __m512i *, const __m512i *);
void sha256_16x32_final_rounds(__m512i *, const __m512i *, const __m512i *, const __m512i *,
                               const __m512i *);
int sha256_16x32_transform_le_short(__m512i *, const __m512i *, const __m512i *, const uint32_t *);
void sha256_8x32_prehash_3rounds(__m256i *, __m256i *, const __m256i *, const __m256i *);
void sha256_8x32_final_rounds(__m256i *, const __m256i *, const __m256i *, const __m256i *,
                              const __m256i *);
int sha256_8x32_transform_le_short(__m256i *, const __m256i *, const __m256i *, const uint32_t *);

/* Rare path: re-hash the lanes of a vector that has a candidate. */
static void check_lanes(const fbm_job *job, uint32_t version, uint32_t n, uint64_t left,
                        unsigned lanes, fbm_hits *out)
{
    uint8_t hdr[FBM_HEADER_LEN], hash[32];
    memcpy(hdr, job->header, sizeof hdr);
    fbm_header_set(hdr, FBM_OFF_VERSION, version);
    for (unsigned l = 0; l < lanes && l < left; l++) {
        fbm_header_set(hdr, FBM_OFF_NONCE, n + l);
        fbm_sha256d(hdr, sizeof hdr, hash);
        if (fbm_top32_le(hash) <= job->t7)
            fbm_hits_push(out, version, n + l);
    }
}

#define DEFINE_SCAN(NAME, LANES, V, SET1, ADD, IOTA, BSWAP, PRE, FIN, SHORT)                      \
    uint64_t NAME(const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0, uint64_t nn,         \
                  fbm_hits *out)                                                                  \
    {                                                                                             \
        const uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);                       \
        const uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0xffffffffu, job->t7};                      \
        V mstate1[8], mstate2[8], mexp_pre[8], istate[8], hash32[8];                              \
        V buf[16], block[16];                                                                     \
        for (uint32_t r = r0; r < r0 + nr; r++) {                                                 \
            const uint32_t version = fbm_rolled_version(base, r);                                 \
            uint32_t in[11];                                                                      \
            fbm_job_inputs(job->header, version, in);                                             \
            for (int i = 0; i < 8; i++) {                                                         \
                mstate1[i] = SET1(in[i]);                                                         \
                istate[i] = SET1(fbm_sha256_iv[i]);                                               \
            }                                                                                     \
            V nv = ADD(SET1(n0), IOTA);                                                           \
            buf[0] = SET1(in[8]);                                                                 \
            buf[1] = SET1(in[9]);                                                                 \
            buf[2] = SET1(in[10]);                                                                \
            buf[3] = BSWAP(nv);                                                                   \
            buf[4] = SET1(0x80000000u);                                                           \
            for (int i = 5; i < 15; i++)                                                          \
                buf[i] = SET1(0);                                                                 \
            buf[15] = SET1(80 * 8);                                                               \
            PRE(mstate2, mexp_pre, buf, mstate1);                                                 \
            block[8] = SET1(0x80000000u);                                                         \
            for (int i = 9; i < 15; i++)                                                          \
                block[i] = SET1(0);                                                               \
            block[15] = SET1(32 * 8);                                                             \
            for (uint64_t i = 0; i < nn; i += LANES) {                                            \
                buf[3] = BSWAP(nv);                                                               \
                FIN(block, buf, mstate1, mstate2, mexp_pre);                                      \
                if (__builtin_expect(SHORT(hash32, block, istate, target), 0))                    \
                    check_lanes(job, version, (uint32_t)(n0 + i), nn - i, LANES, out);            \
                nv = ADD(nv, SET1(LANES));                                                        \
            }                                                                                     \
        }                                                                                         \
        return (uint64_t)nr * nn;                                                                 \
    }

#define SET1_16(x) _mm512_set1_epi32((int)(x))
#define ADD_16(a, b) _mm512_add_epi32((a), (b))
#define IOTA_16 _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
#define BSWAP_16(x) \
    _mm512_shuffle_epi8((x), _mm512_set4_epi32(0x0c0d0e0f, 0x08090a0b, 0x04050607, 0x00010203))
DEFINE_SCAN(fbm_scan_cpuminer_opt16, 16, __m512i, SET1_16, ADD_16, IOTA_16, BSWAP_16,
            sha256_16x32_prehash_3rounds, sha256_16x32_final_rounds,
            sha256_16x32_transform_le_short)

#define SET1_8(x) _mm256_set1_epi32((int)(x))
#define ADD_8(a, b) _mm256_add_epi32((a), (b))
#define IOTA_8 _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
#define BSWAP_8(x) _mm256_shuffle_epi8((x), _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, \
    8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12))
DEFINE_SCAN(fbm_scan_cpuminer_opt8, 8, __m256i, SET1_8, ADD_8, IOTA_8, BSWAP_8,
            sha256_8x32_prehash_3rounds, sha256_8x32_final_rounds, sha256_8x32_transform_le_short)
