/* T1-T4 (midstate, precomputation, early exit) in plain scalar C, plus the
 * same with version rolling so one block-2 schedule serves many versions. */
#include "kern_common.h"

#define LANES 1
#define V uint32_t
#define V_ADD(a, b) ((uint32_t)((a) + (b)))
#define V_XOR(a, b) ((a) ^ (b))
#define V_AND(a, b) ((a) & (b))
#define V_SET1(x) ((uint32_t)(x))
#define V_BSIG0 S_BSIG0
#define V_BSIG1 S_BSIG1
#define V_SSIG0 S_SSIG0
#define V_SSIG1 S_SSIG1
#define V_CH S_CH
#define V_MAJ S_MAJ
#define V_IOTA 0u
#define V_BSWAP(x) fbm_bswap32(x)
#define V_LE_MASK(a, b) ((unsigned)((a) <= (b)))

#include "gen/g_scalar_nonce.h"
#include "gen/g_scalar_vr.h"

#define GEN_NONCE g_scalar_nonce
#define GEN_VR g_scalar_vr
#define NJ_NONCE G_SCALAR_NONCE_NJ
#define NJ_VR G_SCALAR_VR_NJ
#define NL_VR G_SCALAR_VR_NL
#define NN_VR G_SCALAR_VR_NN
#define SCAN_NONCE fbm_scan_scalar
#define SCAN_VR fbm_scan_scalar_vr
#include "kern_template.h"
