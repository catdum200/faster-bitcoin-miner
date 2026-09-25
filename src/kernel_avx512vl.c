/* 8 lanes per 256-bit vector, but using AVX-512VL's vprord and vpternlogd.
 * Same instruction count per lane-round as the 512-bit kernel; 256-bit
 * integer ops can also use port 1, and avoid 512-bit frequency licenses. */
#include <immintrin.h>

#include "kern_common.h"

#define LANES 8
#define V __m256i
#define V_ADD(a, b) _mm256_add_epi32((a), (b))
#define V_XOR(a, b) _mm256_xor_si256((a), (b))
#define V_AND(a, b) _mm256_and_si256((a), (b))
#define V_SET1(x) _mm256_set1_epi32((int)(x))
#define V_XOR3(a, b, c) _mm256_ternarylogic_epi32((a), (b), (c), 0x96)
#define V_CH(e, f, g) _mm256_ternarylogic_epi32((e), (f), (g), 0xca)
#define V_MAJ(a, b, c) _mm256_ternarylogic_epi32((a), (b), (c), 0xe8)
#define V_BSIG0(x) V_XOR3(_mm256_ror_epi32((x), 2), _mm256_ror_epi32((x), 13), _mm256_ror_epi32((x), 22))
#define V_BSIG1(x) V_XOR3(_mm256_ror_epi32((x), 6), _mm256_ror_epi32((x), 11), _mm256_ror_epi32((x), 25))
#define V_SSIG0(x) V_XOR3(_mm256_ror_epi32((x), 7), _mm256_ror_epi32((x), 18), _mm256_srli_epi32((x), 3))
#define V_SSIG1(x) V_XOR3(_mm256_ror_epi32((x), 17), _mm256_ror_epi32((x), 19), _mm256_srli_epi32((x), 10))
#define V_IOTA _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
#define V_BSWAP(x) _mm256_shuffle_epi8((x), _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, \
    15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12))
#define V_LE_MASK(a, b) ((unsigned)_mm256_cmple_epu32_mask((a), (b)))

/* Same generated code as the 512-bit kernel: only the vector width differs. */
#include "gen/g_avx512_nonce.h"
#include "gen/g_avx512_vr.h"

#define GEN_NONCE g_avx512_nonce
#define GEN_VR g_avx512_vr
#define NJ_NONCE G_AVX512_NONCE_NJ
#define NJ_VR G_AVX512_VR_NJ
#define NL_VR G_AVX512_VR_NL
#define NN_VR G_AVX512_VR_NN
#define SCAN_NONCE fbm_scan_avx512vl
#define SCAN_VR fbm_scan_avx512vl_vr
#include "kern_template.h"
