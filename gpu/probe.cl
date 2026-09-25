/* Issue-rate probes: the GPU analogue of `fbm freq`. Each kernel runs 8
 * independent chains of exactly one instruction (inline asm on AMD GPUs), 64
 * instructions per loop iteration, so its throughput shows whether that
 * instruction is full rate. `v_add_nc_u32` is the full-rate reference: its
 * rate / lanes estimates the shader clock under this load.
 *
 * Two mixes answer the open question behind the vr kernel's 1.10-1.17x
 * range: do scalar-ALU and s_delay_alu instructions take the SIMD's issue
 * slot? probe_mix_salu adds 64 s_xor_b32 and probe_mix_delay 64 s_delay_alu
 * to the 64 v_add_nc_u32 of probe_add. Same time as probe_add: free. Double:
 * they cost a slot each.
 *
 * On other devices (e.g. PoCL on a CPU) the plain-C fallbacks run, which
 * only checks that the code path works. */

#if defined(__AMDGCN__) && !defined(FBM_PROBE_PLAIN)
#define A(asmtext, x, y) __asm__ volatile(asmtext : "+v"(x) : "v"(y))
#define A3(asmtext, x, y) __asm__ volatile(asmtext : "+v"(x) : "v"(y), "v"(x))
#define V_ADD(x, y) A("v_add_nc_u32 %0, %0, %1", x, y)
#define V_ALIGNBIT(x, y) A("v_alignbit_b32 %0, %0, %1, 7", x, y)
#define V_XOR3(x, y) A3("v_xor3_b32 %0, %0, %1, %2", x, y)
#define V_ADD3(x, y) A3("v_add3_u32 %0, %0, %1, %2", x, y)
#define V_BFI(x, y) A3("v_bfi_b32 %0, %0, %1, %2", x, y)
#define V_XAD(x, y) A3("v_xad_u32 %0, %0, %1, %2", x, y)
#define S_XOR(s, t) __asm__ volatile("s_xor_b32 %0, %0, %1" : "+s"(s) : "s"(t))
#define S_DELAY() __asm__ volatile("s_delay_alu 0")
#else
#define V_ADD(x, y) (x) = (x) + (y)
#define V_ALIGNBIT(x, y) (x) = ((x) << 25) | ((y) >> 7)
#define V_XOR3(x, y) (x) = (x) ^ (y) ^ ((x) >> 1)
#define V_ADD3(x, y) (x) = (x) + (y) + ((x) >> 1)
#define V_BFI(x, y) (x) = ((x) & (y)) | (~(x) & ((y) >> 1))
#define V_XAD(x, y) (x) = ((x) ^ (y)) + ((x) >> 1)
#define S_XOR(s, t) (s) = (s) ^ (t)
#define S_DELAY()
#endif

/* 8 chains; chain i is combined with chain i+1 of the previous step, so no
 * two instructions in a row depend on each other. */
#define STEP(OP) \
    OP(x0, x1); OP(x1, x2); OP(x2, x3); OP(x3, x4); OP(x4, x5); OP(x5, x6); OP(x6, x7); OP(x7, x0);
#define STEP_SALU(OP) \
    OP(x0, x1); S_XOR(s0, t); OP(x1, x2); S_XOR(s1, t); OP(x2, x3); S_XOR(s2, t); \
    OP(x3, x4); S_XOR(s3, t); OP(x4, x5); S_XOR(s0, t); OP(x5, x6); S_XOR(s1, t); \
    OP(x6, x7); S_XOR(s2, t); OP(x7, x0); S_XOR(s3, t);
#define STEP_DELAY(OP) \
    OP(x0, x1); S_DELAY(); OP(x1, x2); S_DELAY(); OP(x2, x3); S_DELAY(); OP(x3, x4); S_DELAY(); \
    OP(x4, x5); S_DELAY(); OP(x5, x6); S_DELAY(); OP(x6, x7); S_DELAY(); OP(x7, x0); S_DELAY();

#define PROBE(name, STEPX, OP)                                                               \
    __kernel void name(uint iters, uint seed, __global uint *out)                            \
    {                                                                                        \
        const uint id = get_local_id(0) + get_group_id(0) * get_local_size(0);               \
        uint x0 = id ^ seed, x1 = x0 * 3u, x2 = x0 * 5u, x3 = x0 * 7u, x4 = x0 * 11u,        \
             x5 = x0 * 13u, x6 = x0 * 17u, x7 = x0 * 19u;                                     \
        uint s0 = seed, s1 = seed * 3u, s2 = seed * 5u, s3 = seed * 7u, t = seed | 1u;       \
        for (uint i = 0; i < iters; i++) {                                                   \
            STEPX(OP) STEPX(OP) STEPX(OP) STEPX(OP) STEPX(OP) STEPX(OP) STEPX(OP) STEPX(OP) \
        }                                                                                    \
        const uint r = x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7 ^ s0 ^ s1 ^ s2 ^ s3;             \
        if (r == 0x9e3779b9u) /* practically never; keeps the work alive */                 \
            out[0] = r;                                                                      \
    }

PROBE(probe_add, STEP, V_ADD)
PROBE(probe_alignbit, STEP, V_ALIGNBIT)
PROBE(probe_xor3, STEP, V_XOR3)
PROBE(probe_add3, STEP, V_ADD3)
PROBE(probe_bfi, STEP, V_BFI)
PROBE(probe_xad, STEP, V_XAD)
PROBE(probe_mix_salu, STEP_SALU, V_ADD)
PROBE(probe_mix_delay, STEP_DELAY, V_ADD)
