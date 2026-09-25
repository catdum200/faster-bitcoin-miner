/* The generated GPU (RDNA) code evaluated one lane at a time in C. The host
 * uses it for the per-job and per-version precompute (_job, _lane), and the
 * rdna-emu CPU kernels run the whole body with it. */
#ifndef FBM_RDNA_SCALAR_H
#define FBM_RDNA_SCALAR_H

#include "kern_common.h"

#define LANES 1
#define V uint32_t
#define V_ADD(a, b) ((uint32_t)((a) + (b)))
#define V_ADD3(a, b, c) ((uint32_t)((a) + (b) + (c)))
#define V_XOR(a, b) ((a) ^ (b))
#define V_SET1(x) ((uint32_t)(x))
#define V_BFI S_BFI
#define V_BSIG0 S_BSIG0
#define V_BSIG1 S_BSIG1
#define V_SSIG0 S_SSIG0
#define V_SSIG1 S_SSIG1
#define V_CH S_CH
#define V_IOTA 0u
#define V_BSWAP(x) fbm_bswap32(x)
#define V_LE_MASK(a, b) ((unsigned)((a) <= (b)))

#include "gen/g_rdna_nonce.h"
#include "gen/g_rdna_vr.h"

#endif
