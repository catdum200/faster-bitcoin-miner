/* OpenCL C prelude for the generated SHA-256d code (src/gen/g_rdna_*.h).
 *
 * The host builds one program from four strings, in this order:
 *   gpu/prelude.cl, src/gen/g_rdna_nonce.h, src/gen/g_rdna_vr.h, gpu/kernels.cl
 * Each work-item is one SIMD lane, so the generated "vector" type V is a
 * plain uint. Everything is written in portable OpenCL C 1.2 expressions that
 * the AMD compiler maps to single RDNA instructions:
 *   rotate            v_alignbit_b32
 *   a ^ b ^ c         v_xor3_b32
 *   a + b + c         v_add3_u32
 *   BFI(m, x, y)      v_bfi_b32   (Ch, and the second half of Maj)
 *   BSWAP             v_perm_b32
 * The same text is compiled with clang for gfx1200 by tools/gpu_isacheck.py
 * (FBM_AUDIT: no OpenCL library there, so a few builtins are mapped). */

/* A macro, not a typedef: some OpenCL compilers already declare uint32_t. */
#define uint32_t uint

#ifdef FBM_AUDIT
#define get_local_id(d) __builtin_amdgcn_workitem_id_x()
#define get_group_id(d) ((d) == 0 ? __builtin_amdgcn_workgroup_id_x() : __builtin_amdgcn_workgroup_id_y())
#define atomic_inc(p) __atomic_fetch_add((p), 1u, __ATOMIC_RELAXED)
#endif

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define BFI(m, x, y) ((y) ^ ((m) & ((x) ^ (y))))
#define BSWAP(x) (((x) << 24) | (((x) << 8) & 0x00ff0000u) | (((x) >> 8) & 0x0000ff00u) | ((x) >> 24))

/* Scalar precompute sections (the schedule pre-pass uses _nonce). */
#define S_BSIG0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S_BSIG1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define S_SSIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define S_SSIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))
#define S_CH(e, f, g) BFI(e, f, g)
#define S_BFI(m, x, y) BFI(m, x, y)

/* The per-lane body. */
#define V uint
#define V_SET1(x) ((uint)(x))
#define V_ADD(a, b) ((a) + (b))
#define V_ADD3(a, b, c) ((a) + (b) + (c))
#define V_XOR(a, b) ((a) ^ (b))
#define V_BFI(m, x, y) BFI(m, x, y)
#define V_CH(e, f, g) BFI(e, f, g)
#define V_BSIG0 S_BSIG0
#define V_BSIG1 S_BSIG1
#define V_SSIG0 S_SSIG0
#define V_SSIG1 S_SSIG1
