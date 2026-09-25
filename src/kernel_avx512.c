/* 16 lanes per 512-bit vector. AVX-512 has a native rotate (vprord) and
 * vpternlogd, which does Ch, Maj or a three-way XOR in one instruction. */
#include <immintrin.h>

#include "kern_common.h"

#define LANES 16
#define V __m512i
#define V_ADD(a, b) _mm512_add_epi32((a), (b))
#define V_XOR(a, b) _mm512_xor_si512((a), (b))
#define V_AND(a, b) _mm512_and_si512((a), (b))
#define V_SET1(x) _mm512_set1_epi32((int)(x))
#define V_XOR3(a, b, c) _mm512_ternarylogic_epi32((a), (b), (c), 0x96)
#define V_CH(e, f, g) _mm512_ternarylogic_epi32((e), (f), (g), 0xca)
#define V_MAJ(a, b, c) _mm512_ternarylogic_epi32((a), (b), (c), 0xe8)
#define V_BSIG0(x) V_XOR3(_mm512_ror_epi32((x), 2), _mm512_ror_epi32((x), 13), _mm512_ror_epi32((x), 22))
#define V_BSIG1(x) V_XOR3(_mm512_ror_epi32((x), 6), _mm512_ror_epi32((x), 11), _mm512_ror_epi32((x), 25))
#define V_SSIG0(x) V_XOR3(_mm512_ror_epi32((x), 7), _mm512_ror_epi32((x), 18), _mm512_srli_epi32((x), 3))
#define V_SSIG1(x) V_XOR3(_mm512_ror_epi32((x), 17), _mm512_ror_epi32((x), 19), _mm512_srli_epi32((x), 10))
#define V_IOTA _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
#define V_BSWAP(x) _mm512_shuffle_epi8((x), _mm512_set4_epi32(0x0c0d0e0f, 0x08090a0b, 0x04050607, 0x00010203))
#define V_LE_MASK(a, b) ((unsigned)_mm512_cmple_epu32_mask((a), (b)))

#include "gen/g_avx512_nonce.h"
#include "gen/g_avx512_vr.h"

#define GEN_NONCE g_avx512_nonce
#define GEN_VR g_avx512_vr
#define NJ_NONCE G_AVX512_NONCE_NJ
#define NJ_VR G_AVX512_VR_NJ
#define NL_VR G_AVX512_VR_NL
#define NN_VR G_AVX512_VR_NN
#define SCAN_NONCE fbm_scan_avx512
#define SCAN_VR fbm_scan_avx512_vr
#include "kern_template.h"
