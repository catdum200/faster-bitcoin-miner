/* The GPU (AMD RDNA) generated kernels, run one lane at a time on the CPU.
 * They are not for mining: they put the exact generated code that the
 * OpenCL kernels run (src/gen/g_rdna_*.h) through every CPU correctness
 * test, including on machines without a GPU. */
#include "rdna_scalar.h"

#define GEN_NONCE g_rdna_nonce
#define GEN_VR g_rdna_vr
#define NJ_NONCE G_RDNA_NONCE_NJ
#define NJ_VR G_RDNA_VR_NJ
#define NL_VR G_RDNA_VR_NL
#define NN_VR G_RDNA_VR_NN
#define SCAN_NONCE fbm_scan_rdna_emu
#define SCAN_VR fbm_scan_rdna_emu_vr
#include "kern_template.h"
