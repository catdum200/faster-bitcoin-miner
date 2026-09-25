/* 8 lanes per 256-bit AVX2 vector. AVX2 has no vector rotate and no
 * three-input logic op, so each rotate costs a shift, a shift and an OR. */
#include <immintrin.h>

#include "kern_common.h"

#define LANES 8
#define V __m256i
#define V_ADD(a, b) _mm256_add_epi32((a), (b))
#define V_XOR(a, b) _mm256_xor_si256((a), (b))
#define V_AND(a, b) _mm256_and_si256((a), (b))
#define V_SET1(x) _mm256_set1_epi32((int)(x))
#define V_ROR(x, n) _mm256_or_si256(_mm256_srli_epi32((x), (n)), _mm256_slli_epi32((x), 32 - (n)))
#define V_BSIG0(x) V_XOR(V_XOR(V_ROR(x, 2), V_ROR(x, 13)), V_ROR(x, 22))
#define V_BSIG1(x) V_XOR(V_XOR(V_ROR(x, 6), V_ROR(x, 11)), V_ROR(x, 25))
#define V_SSIG0(x) V_XOR(V_XOR(V_ROR(x, 7), V_ROR(x, 18)), _mm256_srli_epi32((x), 3))
#define V_SSIG1(x) V_XOR(V_XOR(V_ROR(x, 17), V_ROR(x, 19)), _mm256_srli_epi32((x), 10))
#define V_IOTA _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
#define V_BSWAP(x) _mm256_shuffle_epi8((x), _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, \
    15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12))
/* a <= b (unsigned) iff min(a, b) == a */
#define V_LE_MASK(a, b) ((unsigned)_mm256_movemask_ps(_mm256_castsi256_ps( \
    _mm256_cmpeq_epi32(_mm256_min_epu32((a), (b)), (a)))))

#include "gen/g_avx2_nonce.h"
#include "gen/g_avx2_vr.h"

#define GEN_NONCE g_avx2_nonce
#define GEN_VR g_avx2_vr
#define NJ_NONCE G_AVX2_NONCE_NJ
#define NJ_VR G_AVX2_VR_NJ
#define NL_VR G_AVX2_VR_NL
#define NN_VR G_AVX2_VR_NN
#define SCAN_NONCE fbm_scan_avx2
#define SCAN_VR fbm_scan_avx2_vr
#include "kern_template.h"
