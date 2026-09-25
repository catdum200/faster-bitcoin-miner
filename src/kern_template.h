/* Instantiates two scan functions around a pair of generated kernels.
 *
 * The including file defines:
 *   LANES, V and the V_* ops used by the generated body,
 *   V_IOTA        vector {0, 1, ..., LANES-1}
 *   V_BSWAP(x)    byte-swap every 32-bit lane
 *   V_LE_MASK(a, b) bitmask of lanes with a <= b (unsigned)
 *   GEN_NONCE / GEN_VR  generated-function prefixes (nonce-lane / version-lane)
 *   NJ_NONCE, NJ_VR, NL_VR, NN_VR  slot counts from the generated headers
 *   SCAN_NONCE / SCAN_VR  names of the scan functions to define
 */
#define FBM_CAT_(a, b) a##b
#define FBM_CAT(a, b) FBM_CAT_(a, b)

/* SCAN_NONCE: every lane hashes the same version with a different nonce. */
uint64_t SCAN_NONCE(const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0, uint64_t nn,
                    fbm_hits *out)
{
    const uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);
    const V t7 = V_SET1(job->t7);

    for (uint32_t r = r0; r < r0 + nr; r++) {
        const uint32_t version = fbm_rolled_version(base, r);
        uint32_t in[11], J[NJ_NONCE];

        fbm_job_inputs(job->header, version, in);
        FBM_CAT(GEN_NONCE, _job)(in, J);
        for (uint64_t i = 0; i < nn; i += LANES) {
            const V nonce = V_ADD(V_SET1((uint32_t)(n0 + i)), V_IOTA);
            const V h7 = FBM_CAT(GEN_NONCE, _body)(J, NULL, NULL, V_BSWAP(nonce));
            unsigned m = V_LE_MASK(V_BSWAP(h7), t7);
            while (m) {
                const unsigned l = (unsigned)__builtin_ctz(m);
                m &= m - 1;
                if (i + l < nn)
                    fbm_hits_push(out, version, (uint32_t)(n0 + i + l));
            }
        }
    }
    return (uint64_t)nr * nn;
}

/* SCAN_VR: every lane hashes a different rolled version (its own midstate)
 * with the same nonce. The block-2 message schedule depends only on the
 * nonce, so it is computed once per nonce in scalar code and shared by
 * every lane and every group of versions (overt ASICBoost, BIP 320). */
#define VR_GROUPS 8

static __attribute__((noinline)) void FBM_CAT(GEN_VR, _nonce_ni)(uint32_t w3, const uint32_t *J,
                                                                 uint32_t *N)
{
    FBM_CAT(GEN_VR, _nonce)(w3, J, N);
}

uint64_t SCAN_VR(const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0, uint64_t nn,
                 fbm_hits *out)
{
    const uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);
    const V t7 = V_SET1(job->t7);
    uint32_t in[11], J[NJ_VR];

    /* The per-job words (W0..W2) do not depend on the version. */
    fbm_job_inputs(job->header, base, in);
    FBM_CAT(GEN_VR, _job)(in, J);

    for (uint32_t rc = 0; rc < nr; rc += LANES * VR_GROUPS) {
        static _Thread_local uint32_t Lraw[VR_GROUPS][NL_VR][LANES] __attribute__((aligned(64)));
        uint32_t versions[VR_GROUPS][LANES];
        uint32_t left = nr - rc;
        const uint32_t groups = left >= LANES * VR_GROUPS ? VR_GROUPS : (left + LANES - 1) / LANES;

        for (uint32_t g = 0; g < groups; g++) {
            for (uint32_t l = 0; l < LANES; l++) {
                const uint32_t idx = rc + g * LANES + l;
                /* Lanes past the end repeat version r0; their hits are dropped. */
                const uint32_t version = fbm_rolled_version(base, r0 + (idx < nr ? idx : 0));
                uint32_t lin[11], Ls[NL_VR];
                fbm_job_inputs(job->header, version, lin);
                FBM_CAT(GEN_VR, _lane)(lin, J, Ls);
                for (int k = 0; k < NL_VR; k++)
                    Lraw[g][k][l] = Ls[k];
                versions[g][l] = version;
            }
        }
        for (uint64_t i = 0; i < nn; i++) {
            const uint32_t nonce = (uint32_t)(n0 + i);
            uint32_t N[NN_VR] __attribute__((aligned(64)));
            FBM_CAT(GEN_VR, _nonce_ni)(fbm_bswap32(nonce), J, N);
            for (uint32_t g = 0; g < groups; g++) {
                const V h7 = FBM_CAT(GEN_VR, _body)(J, (const V *)Lraw[g], N, V_SET1(0));
                unsigned m = V_LE_MASK(V_BSWAP(h7), t7);
                while (m) {
                    const unsigned l = (unsigned)__builtin_ctz(m);
                    m &= m - 1;
                    if (rc + g * LANES + l < nr)
                        fbm_hits_push(out, versions[g][l], nonce);
                }
            }
        }
    }
    return (uint64_t)nr * nn;
}
