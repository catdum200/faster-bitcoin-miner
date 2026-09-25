/* Correctness tests. The reference kernel (portable, byte-oriented SHA-256
 * over the whole header) is the oracle; every optimized kernel must report
 * exactly the same candidate set on the same rectangle of the search space. */
#include "selftest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/sha.h>

#include "header.h"
#include "kernels.h"
#include "miner.h"
#include "sha256.h"
#include "util.h"

static int failures;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            failures++;                                   \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
        }                                                 \
    } while (0)

/* Real mainnet headers: genesis (height 0) and height 125552. */
const char *const fbm_genesis_hex =
    "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27a"
    "c72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c";
static const char *const genesis_hash =
    "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";
const char *const fbm_block125552_hex =
    "0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000e320b6c2fffc8d75"
    "0423db8b1eb942ae710e951ed797f7affc8892b0f1fc122bc7f5d74df2b9441a42a14695";
static const char *const block125552_hash =
    "00000000000000001e8d6829a8a21adc5d38d0a473b144b6765798e61f98bd1d";

static void hex_sha256(const char *msg, size_t len, char out[65])
{
    uint8_t h[32];
    fbm_sha256(msg, len, h);
    fbm_hex_encode(h, 32, out);
}

static void test_sha256(void)
{
    char hex[65];
    printf("sha256 FIPS 180 vectors\n");
    hex_sha256("", 0, hex);
    CHECK(!strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "empty: %s", hex);
    hex_sha256("abc", 3, hex);
    CHECK(!strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "abc: %s", hex);
    const char *m448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    hex_sha256(m448, strlen(m448), hex);
    CHECK(!strcmp(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"), "448-bit: %s", hex);

    /* One million 'a' through the streaming API in uneven pieces. */
    fbm_sha256_ctx ctx;
    uint8_t h[32];
    char chunk[997];
    memset(chunk, 'a', sizeof chunk);
    fbm_sha256_init(&ctx);
    size_t left = 1000000;
    while (left) {
        size_t n = left < sizeof chunk ? left : sizeof chunk;
        fbm_sha256_update(&ctx, chunk, n);
        left -= n;
    }
    fbm_sha256_final(&ctx, h);
    fbm_hex_encode(h, 32, hex);
    CHECK(!strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"), "1M a: %s", hex);
}

static void test_blocks(void)
{
    const struct {
        const char *name, *hex, *hash;
        double difficulty;
    } blocks[] = {
        {"genesis", fbm_genesis_hex, genesis_hash, 1.0},
        {"125552", fbm_block125552_hex, block125552_hash, 244112.48777434},
    };
    printf("real block headers\n");
    for (size_t i = 0; i < sizeof blocks / sizeof blocks[0]; i++) {
        uint8_t hdr[80], hash[32], target[32];
        char disp[65];
        CHECK(fbm_hex_decode(blocks[i].hex, hdr, 80) == 0, "%s: hex decode", blocks[i].name);
        fbm_sha256d(hdr, 80, hash);
        fbm_hash_display(hash, disp);
        CHECK(!strcmp(disp, blocks[i].hash), "%s: hash %s", blocks[i].name, disp);
        CHECK(fbm_bits_to_target(fbm_header_get(hdr, FBM_OFF_BITS), target) == 0, "%s: bits", blocks[i].name);
        CHECK(fbm_cmp256_le(hash, target) <= 0, "%s: hash above target", blocks[i].name);
        double d = fbm_target_difficulty(target);
        CHECK(d > blocks[i].difficulty * 0.999999 && d < blocks[i].difficulty * 1.000001,
              "%s: difficulty %.8f", blocks[i].name, d);
    }
    uint8_t t[32];
    CHECK(fbm_bits_to_target(0x1d00ffff, t) == 0 && t[26] == 0xff && t[27] == 0xff && t[28] == 0 &&
              fbm_top32_le(t) == 0, "diff-1 target layout");
    CHECK(fbm_bits_to_target(0x04123456, t) == 0 && t[1] == 0x56 && t[2] == 0x34 && t[3] == 0x12 &&
              t[0] == 0, "compact 0x04123456 layout");
    CHECK(fbm_bits_to_target(0x04923456, t) != 0, "negative compact (sign bit) rejected");
    CHECK(fbm_bits_to_target(0x21010000, t) != 0, "overflowing compact rejected");
    CHECK(fbm_bits_to_target(0x207fffff, t) == 0 && fbm_top32_le(t) == 0x7fffff00, "regtest target");
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint32_t rng32(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545f4914f6cdd1dull) >> 32);
}

static int hit_cmp(const void *a, const void *b)
{
    const fbm_hit *x = a, *y = b;
    if (x->version != y->version)
        return x->version < y->version ? -1 : 1;
    if (x->nonce != y->nonce)
        return x->nonce < y->nonce ? -1 : 1;
    return 0;
}

static fbm_hits run(const fbm_kernel *k, const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                    uint64_t nn)
{
    fbm_hits h = {calloc(1u << 16, sizeof(fbm_hit)), 1u << 16, 0};
    uint64_t done = k->scan(job, r0, nr, n0, nn, &h);
    CHECK(done == (uint64_t)nr * nn, "%s: returned %llu hashes", k->name, (unsigned long long)done);
    size_t stored = h.n < h.cap ? h.n : h.cap;
    qsort(h.v, stored, sizeof(fbm_hit), hit_cmp);
    return h;
}

/* Compare every supported kernel with the reference on one rectangle. */
static void diff_case(const char *label, const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                      uint64_t nn)
{
    const fbm_kernel *ref = fbm_kernel_find("ref");
    fbm_hits want = run(ref, job, r0, nr, n0, nn);
    CHECK(want.n <= want.cap, "%s: too many reference hits", label);
    for (size_t i = 0; i < fbm_kernel_count(); i++) {
        const fbm_kernel *k = fbm_kernel_at(i);
        if (k == ref || !k->supported())
            continue;
        fbm_hits got = run(k, job, r0, nr, n0, nn);
        int same = got.n == want.n && !memcmp(got.v, want.v, want.n * sizeof(fbm_hit));
        CHECK(same, "%s: kernel %s found %zu candidates, reference %zu", label, k->name, got.n,
              want.n);
        free(got.v);
    }
    free(want.v);
}

static void random_job(fbm_job *job, uint32_t t7)
{
    for (int i = 0; i < 80; i++)
        job->header[i] = (uint8_t)rng32();
    job->t7 = t7;
}

static void test_kernels_differential(void)
{
    fbm_job job;
    printf("kernels vs reference (random headers)\n");
    random_job(&job, 0x0fffffff); /* ~1/16 of hashes are candidates */
    diff_case("loose", &job, 0, 32, 1000, 4096);
    random_job(&job, 0x000fffff); /* ~1/4096 */
    diff_case("tight", &job, 7, 48, 123456789, 16384);
    random_job(&job, 0xfffffffe);
    diff_case("nearly all", &job, 0, 16, 0, 64);
    random_job(&job, 0x3fffffff);
    diff_case("odd sizes", &job, 3, 5, 77, 37);
    random_job(&job, 0x3fffffff);
    diff_case("end of nonce space", &job, FBM_VR_MAX - 17, 17, 0xffffffffu - 100, 101);
    random_job(&job, 0);
    diff_case("production t7=0", &job, 0, 16, 0, 1 << 14);
}

/* Mainnet blocks whose versions have BIP 320 bits rolled by the ASIC that
 * mined them: heights 800000-910000 and three from September 2026 (ViaBTC,
 * Braiins Pool, MARA Pool). Hashes were checked against mempool.space. */
static const char *const recent_blocks[][2] = {
    {"800000", "00601d3455bb9fbd966b3ea2dc42d0c22722e4c0c1729fad172101000000000000000000550"
               "87fab0c8f3f89f8bcfd4df26c504d81b0a88e04907161838c0c53001af09135edbd64943805175e955e06"},
    {"850000", "0080bb2b13b3152752d9cf2a36fcd16d78f64269d847932f076b02000000000000000000d51"
               "a6bd669cf6bc30269a259c2918e6319269fc15f020a82ddd6a5fdc1f5cf71ca618066255d03176ab0fb7c"},
    {"900000", "00a0ab20247d4d9f582f9750344cdf62c46d81d046be960340960100000000000000000070f"
               "96945530651135839d8adc3f40e595118ec74c7ad81a3d17bb022e554fb0c937f4268743702177ad05f92"},
    {"910000", "00a0572be06d4f01a2ed2228dec965539cc8b96512ccde7d2824010000000000000000006f2"
               "8c30dc748f6b1430fb2b9a5a94b5b34a5df6e318c6cc5c310a1a35b432b59a3ab9d68b32c021719d103e9"},
    {"968566", "0000003874fd1f0f0ea2296a90502e569622d8e3bc1f8d201d650000000000000000000001c20b8594f8"
               "5b5e3d92e4a66ac4f2864913fcd8b44c4a3bbf7da256378081ec2fa1b66ac51e02177b22a039"},
    {"968562", "0060aa297152dfd5bda8f97830a47df977bf41a6254159cc0366000000000000000000008abded7270cd"
               "1dfde5a3cfd286b8a1787980c006b9f35b8f47146dc68abdbac0ef95b66ac51e021718227e7a"},
    {"968555", "00607925ad9b791fcca5e6d220cd922225cc79e7811a78d775110000000000000000000033382e41f38a"
               "d364b21b717a3baa7cdf34e834c52bbcd3f800d785aeac3539ff6a7eb66ac51e021748202491"},
};

/* Each kernel must rediscover the real (version, nonce) of real blocks. The
 * job's base version is the real one XOR (5 << 13), so the solution sits at
 * rolled-version index r = 5 of the 16 scanned: this checks the BIP 320
 * version mapping as well as the hashing. */
static void test_kernels_known(void)
{
    const char *names[16], *hexes[16];
    size_t nblocks = 0;
    names[nblocks] = "genesis";
    hexes[nblocks++] = fbm_genesis_hex;
    names[nblocks] = "125552";
    hexes[nblocks++] = fbm_block125552_hex;
    for (size_t i = 0; i < sizeof recent_blocks / sizeof recent_blocks[0]; i++) {
        names[nblocks] = recent_blocks[i][0];
        hexes[nblocks++] = recent_blocks[i][1];
    }
    printf("kernels rediscover real (version, nonce) of %zu mainnet blocks\n", nblocks);
    for (size_t b = 0; b < nblocks; b++) {
        fbm_job job;
        uint8_t target[32], hash[32];
        fbm_hex_decode(hexes[b], job.header, 80);
        fbm_sha256d(job.header, 80, hash);
        fbm_bits_to_target(fbm_header_get(job.header, FBM_OFF_BITS), target);
        CHECK(fbm_cmp256_le(hash, target) <= 0, "%s: header does not meet its target", names[b]);
        job.t7 = fbm_top32_le(target);
        const uint32_t nonce = fbm_header_get(job.header, FBM_OFF_NONCE);
        const uint32_t version = fbm_header_get(job.header, FBM_OFF_VERSION);
        fbm_header_set(job.header, FBM_OFF_VERSION, fbm_rolled_version(version, 5));
        for (size_t i = 0; i < fbm_kernel_count(); i++) {
            const fbm_kernel *k = fbm_kernel_at(i);
            if (!k->supported())
                continue;
            fbm_hits h = run(k, &job, 0, 16, nonce - 3000, 5000);
            int found = 0;
            for (size_t j = 0; j < h.n && j < h.cap; j++) {
                uint8_t hdr[80];
                memcpy(hdr, job.header, 80);
                fbm_header_set(hdr, FBM_OFF_VERSION, h.v[j].version);
                fbm_header_set(hdr, FBM_OFF_NONCE, h.v[j].nonce);
                fbm_sha256d(hdr, 80, hash);
                CHECK(fbm_top32_le(hash) <= job.t7, "%s/%s: bogus candidate", names[b], k->name);
                if (h.v[j].nonce == nonce && h.v[j].version == version)
                    found = 1;
            }
            CHECK(found, "%s: kernel %s missed version 0x%08x nonce %u", names[b], k->name, version,
                  nonce);
            free(h.v);
        }
    }
}

/* fbm_run must cover a range exactly once however it splits it. */
static void test_partition(void)
{
    const char *names[] = {"scalar", "scalar-vr", "avx2", "avx512", "avx512-vr"};
    const fbm_kernel *ref = fbm_kernel_find("ref");
    fbm_job job;
    printf("thread partitioning covers the range exactly once\n");
    random_job(&job, 0x03ffffff); /* ~1/64 */
    fbm_hits want = run(ref, &job, 2, 16, 4000000000u, 20011);
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        const fbm_kernel *k = fbm_kernel_find(names[i]);
        if (!k || !k->supported())
            continue;
        for (int threads = 1; threads <= 7; threads += 3) {
            fbm_hits got = {calloc(1u << 16, sizeof(fbm_hit)), 1u << 16, 0};
            fbm_run_stats st = fbm_run(k, &job, 2, 16, 4000000000u, 20011, threads, &got);
            qsort(got.v, got.n < got.cap ? got.n : got.cap, sizeof(fbm_hit), hit_cmp);
            int same = st.hashes == 16ull * 20011 && got.n == want.n &&
                       !memcmp(got.v, want.v, want.n * sizeof(fbm_hit));
            CHECK(same, "%s with %d threads: %zu candidates, %llu hashes (want %zu)", k->name,
                  threads, got.n, (unsigned long long)st.hashes, want.n);
            free(got.v);
        }
    }
    free(want.v);
}

/* The reference SHA-256 must agree with OpenSSL on every length. */
static void test_sha256_vs_openssl(void)
{
    uint8_t msg[300], a[32], b[32];
    printf("sha256 vs OpenSSL, lengths 0..299\n");
    for (size_t len = 0; len < sizeof msg; len++) {
        for (size_t i = 0; i < len; i++)
            msg[i] = (uint8_t)rng32();
        fbm_sha256(msg, len, a);
        SHA256(msg, len, b);
        CHECK(!memcmp(a, b, 32), "length %zu differs from OpenSSL", len);
    }
}

int fbm_selftest(void)
{
    failures = 0;
    test_sha256();
    test_sha256_vs_openssl();
    test_blocks();
    test_kernels_differential();
    test_kernels_known();
    test_partition();
    printf(failures ? "%d FAILURE(S)\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
