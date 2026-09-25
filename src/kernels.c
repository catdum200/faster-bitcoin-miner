#include "kernels.h"

#include <string.h>

#define SCAN_DECL(name) \
    uint64_t name(const fbm_job *, uint32_t, uint32_t, uint32_t, uint64_t, fbm_hits *)
SCAN_DECL(fbm_scan_ref);
SCAN_DECL(fbm_scan_openssl);
SCAN_DECL(fbm_scan_scalar);
SCAN_DECL(fbm_scan_scalar_vr);
SCAN_DECL(fbm_scan_avx2);
SCAN_DECL(fbm_scan_avx2_vr);
SCAN_DECL(fbm_scan_avx512vl);
SCAN_DECL(fbm_scan_avx512vl_vr);
SCAN_DECL(fbm_scan_avx512);
SCAN_DECL(fbm_scan_avx512_vr);
#ifdef FBM_BASELINES
SCAN_DECL(fbm_scan_cpuminer_opt8);
SCAN_DECL(fbm_scan_cpuminer_opt16);
#endif

static int cpu_any(void)
{
    return 1;
}

static int cpu_avx2(void)
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

static int cpu_avx512(void)
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl");
}

/* Ordered from slowest to fastest expected; fbm_kernel_best() walks it
 * backwards. */
static const fbm_kernel kernels[] = {
#ifdef FBM_BASELINES
    /* Optional prior-art baselines (make BASELINES=1); never chosen as "best". */
    {"cpuminer-opt8", "baseline: cpuminer-opt v26.1 sha256d 8-way (AVX-512VL ymm build)",
     fbm_scan_cpuminer_opt8, cpu_avx512, 1, 0, FBM_ISA_YMM},
    {"cpuminer-opt16", "baseline: cpuminer-opt v26.1 sha256d 16-way AVX-512",
     fbm_scan_cpuminer_opt16, cpu_avx512, 1, 0, FBM_ISA_ZMM},
#endif
    {"ref", "naive: whole 80-byte header hashed twice per nonce (portable C)", fbm_scan_ref,
     cpu_any, 1, 0, FBM_ISA_SCALAR},
    {"openssl", "naive: whole header hashed twice per nonce with OpenSSL SHA256()",
     fbm_scan_openssl, cpu_any, 1, 0, FBM_ISA_YMM},
    {"scalar", "midstate + constant folding + early exit, 1 nonce at a time", fbm_scan_scalar,
     cpu_any, 1, 0, FBM_ISA_SCALAR},
    {"scalar-vr", "scalar + version rolling: one block-2 schedule per nonce for all versions",
     fbm_scan_scalar_vr, cpu_any, 1, 1, FBM_ISA_SCALAR},
    {"avx2", "8 nonces per AVX2 vector", fbm_scan_avx2, cpu_avx2, 1, 0, FBM_ISA_YMM},
    {"avx2-vr", "8 versions per AVX2 vector, schedule shared in scalar", fbm_scan_avx2_vr,
     cpu_avx2, 8, 1, FBM_ISA_YMM},
    {"avx512vl", "8 nonces per 256-bit vector using AVX-512VL rotate/ternlog",
     fbm_scan_avx512vl, cpu_avx512, 1, 0, FBM_ISA_YMM},
    {"avx512vl-vr", "8 versions per 256-bit AVX-512VL vector, schedule shared in scalar",
     fbm_scan_avx512vl_vr, cpu_avx512, 8, 1, FBM_ISA_YMM},
    {"avx512", "16 nonces per AVX-512 vector", fbm_scan_avx512, cpu_avx512, 1, 0, FBM_ISA_ZMM},
    {"avx512-vr", "16 versions per AVX-512 vector, schedule shared in scalar",
     fbm_scan_avx512_vr, cpu_avx512, 16, 1, FBM_ISA_ZMM},
};

size_t fbm_kernel_count(void)
{
    return sizeof kernels / sizeof kernels[0];
}

const fbm_kernel *fbm_kernel_at(size_t i)
{
    return i < fbm_kernel_count() ? &kernels[i] : NULL;
}

const fbm_kernel *fbm_kernel_find(const char *name)
{
    for (size_t i = 0; i < fbm_kernel_count(); i++) {
        if (strcmp(kernels[i].name, name) == 0)
            return &kernels[i];
    }
    return NULL;
}

const fbm_kernel *fbm_kernel_best(uint32_t versions)
{
    const int want_vr = versions >= 64;
    for (size_t i = fbm_kernel_count(); i-- > 0;) {
        if (kernels[i].supported() && kernels[i].vr == want_vr &&
            strncmp(kernels[i].desc, "baseline:", 9) != 0)
            return &kernels[i];
    }
    return &kernels[0];
}
