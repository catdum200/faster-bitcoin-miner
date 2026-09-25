/* Correctness tests. The reference kernel (portable, byte-oriented SHA-256
 * over the whole header) is the oracle; every optimized kernel must report
 * exactly the same candidate set on the same rectangle of the search space. */
#include "selftest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "header.h"
#include "kernels.h"
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

/* Each kernel must rediscover the real nonce (and only valid candidates). */
static void test_kernels_known(void)
{
    const struct {
        const char *name, *hex;
    } blocks[] = {{"genesis", fbm_genesis_hex}, {"125552", fbm_block125552_hex}};
    printf("kernels rediscover real block nonces\n");
    for (size_t b = 0; b < 2; b++) {
        fbm_job job;
        uint8_t target[32];
        fbm_hex_decode(blocks[b].hex, job.header, 80);
        fbm_bits_to_target(fbm_header_get(job.header, FBM_OFF_BITS), target);
        job.t7 = fbm_top32_le(target);
        const uint32_t nonce = fbm_header_get(job.header, FBM_OFF_NONCE);
        const uint32_t version = fbm_header_get(job.header, FBM_OFF_VERSION);
        for (size_t i = 0; i < fbm_kernel_count(); i++) {
            const fbm_kernel *k = fbm_kernel_at(i);
            if (!k->supported())
                continue;
            /* Version-lane kernels scan 16 versions; r = 0 is the real one. */
            uint32_t nr = k->version_lanes > 1 ? 16 : 1;
            fbm_hits h = run(k, &job, 0, nr, nonce - 3000, 5000);
            int found = 0;
            for (size_t j = 0; j < h.n && j < h.cap; j++) {
                uint8_t hdr[80], hash[32];
                memcpy(hdr, job.header, 80);
                fbm_header_set(hdr, FBM_OFF_VERSION, h.v[j].version);
                fbm_header_set(hdr, FBM_OFF_NONCE, h.v[j].nonce);
                fbm_sha256d(hdr, 80, hash);
                CHECK(fbm_top32_le(hash) <= job.t7, "%s/%s: bogus candidate", blocks[b].name, k->name);
                if (h.v[j].nonce == nonce && h.v[j].version == version)
                    found = 1;
            }
            CHECK(found, "%s: kernel %s missed nonce %u", blocks[b].name, k->name, nonce);
            free(h.v);
        }
    }
}

int fbm_selftest(void)
{
    failures = 0;
    test_sha256();
    test_blocks();
    test_kernels_differential();
    test_kernels_known();
    printf(failures ? "%d FAILURE(S)\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
