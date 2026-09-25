/* Core-clock probes. The TSC ticks at a constant 2.8 GHz here, but the core
 * does not: it turbos on scalar code and drops to a lower "license"
 * frequency while running 512-bit instructions. Each probe times a chain of
 * dependent single-cycle adds, so elapsed time = chain length / core clock. */
#include "freq.h"

#include <stdint.h>
#include <time.h>

#define CHAIN 400000000ull /* dependent adds per probe (~0.15 s) */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double fbm_ghz_scalar(void)
{
    uint64_t x = 1;
    double t0 = now();
    for (uint64_t i = 0; i < CHAIN / 8; i++) {
        __asm__ volatile("add %0, %0\n\tadd %0, %0\n\tadd %0, %0\n\tadd %0, %0\n\t"
                         "add %0, %0\n\tadd %0, %0\n\tadd %0, %0\n\tadd %0, %0"
                         : "+r"(x));
    }
    return CHAIN / (now() - t0) / 1e9;
}

__attribute__((target("avx2"))) double fbm_ghz_ymm(void)
{
    double t0 = now();
    __asm__ volatile("vpxor %%ymm0, %%ymm0, %%ymm0" ::: "xmm0");
    for (uint64_t i = 0; i < CHAIN / 8; i++) {
        __asm__ volatile("vpaddd %%ymm0, %%ymm0, %%ymm0\n\tvpaddd %%ymm0, %%ymm0, %%ymm0\n\t"
                         "vpaddd %%ymm0, %%ymm0, %%ymm0\n\tvpaddd %%ymm0, %%ymm0, %%ymm0\n\t"
                         "vpaddd %%ymm0, %%ymm0, %%ymm0\n\tvpaddd %%ymm0, %%ymm0, %%ymm0\n\t"
                         "vpaddd %%ymm0, %%ymm0, %%ymm0\n\tvpaddd %%ymm0, %%ymm0, %%ymm0" ::
                             : "xmm0");
    }
    double s = now() - t0;
    __asm__ volatile("vzeroupper");
    return CHAIN / s / 1e9;
}

__attribute__((target("avx512f"))) double fbm_ghz_zmm(void)
{
    /* Warm up so the core has switched to the 512-bit license. */
    for (int pass = 0; pass < 2; pass++) {
        double t0 = now();
        __asm__ volatile("vpxord %%zmm0, %%zmm0, %%zmm0" ::: "xmm0");
        for (uint64_t i = 0; i < CHAIN / 8; i++) {
            __asm__ volatile("vpaddd %%zmm0, %%zmm0, %%zmm0\n\tvpaddd %%zmm0, %%zmm0, %%zmm0\n\t"
                             "vpaddd %%zmm0, %%zmm0, %%zmm0\n\tvpaddd %%zmm0, %%zmm0, %%zmm0\n\t"
                             "vpaddd %%zmm0, %%zmm0, %%zmm0\n\tvpaddd %%zmm0, %%zmm0, %%zmm0\n\t"
                             "vpaddd %%zmm0, %%zmm0, %%zmm0\n\tvpaddd %%zmm0, %%zmm0, %%zmm0" ::
                                 : "xmm0");
        }
        double s = now() - t0;
        __asm__ volatile("vzeroupper");
        if (pass == 1)
            return CHAIN / s / 1e9;
    }
    return 0;
}
