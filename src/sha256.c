#include "sha256.h"

#include <string.h>

#include "util.h"

const uint32_t fbm_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

const uint32_t fbm_sha256_iv[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

void fbm_sha256_transform_words(uint32_t state[8], const uint32_t m[16])
{
    uint32_t w[64];
    for (int t = 0; t < 16; t++)
        w[t] = m[t];
    for (int t = 16; t < 64; t++) {
        uint32_t s0 = fbm_rotr32(w[t - 15], 7) ^ fbm_rotr32(w[t - 15], 18) ^ (w[t - 15] >> 3);
        uint32_t s1 = fbm_rotr32(w[t - 2], 17) ^ fbm_rotr32(w[t - 2], 19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int t = 0; t < 64; t++) {
        uint32_t S1 = fbm_rotr32(e, 6) ^ fbm_rotr32(e, 11) ^ fbm_rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + fbm_sha256_k[t] + w[t];
        uint32_t S0 = fbm_rotr32(a, 2) ^ fbm_rotr32(a, 13) ^ fbm_rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void fbm_sha256_transform(uint32_t state[8], const uint8_t block[64])
{
    uint32_t m[16];
    for (int i = 0; i < 16; i++)
        m[i] = fbm_load_be32(block + 4 * i);
    fbm_sha256_transform_words(state, m);
}

void fbm_sha256_init(fbm_sha256_ctx *ctx)
{
    memcpy(ctx->state, fbm_sha256_iv, sizeof ctx->state);
    ctx->total = 0;
    ctx->buflen = 0;
}

void fbm_sha256_update(fbm_sha256_ctx *ctx, const void *data, size_t len)
{
    const uint8_t *p = data;
    ctx->total += len;
    while (len > 0) {
        size_t take = 64 - ctx->buflen;
        if (take > len)
            take = len;
        memcpy(ctx->buf + ctx->buflen, p, take);
        ctx->buflen += take;
        p += take;
        len -= take;
        if (ctx->buflen == 64) {
            fbm_sha256_transform(ctx->state, ctx->buf);
            ctx->buflen = 0;
        }
    }
}

void fbm_sha256_final(fbm_sha256_ctx *ctx, uint8_t out[32])
{
    uint64_t bits = ctx->total * 8;
    uint8_t pad[72] = {0x80};
    size_t padlen = (ctx->buflen < 56) ? 56 - ctx->buflen : 120 - ctx->buflen;
    for (int i = 0; i < 8; i++)
        pad[padlen + i] = (uint8_t)(bits >> (56 - 8 * i));
    /* update() would count the padding in total; total is no longer needed. */
    fbm_sha256_update(ctx, pad, padlen + 8);
    for (int i = 0; i < 8; i++)
        fbm_store_be32(out + 4 * i, ctx->state[i]);
}

void fbm_sha256(const void *data, size_t len, uint8_t out[32])
{
    fbm_sha256_ctx ctx;
    fbm_sha256_init(&ctx);
    fbm_sha256_update(&ctx, data, len);
    fbm_sha256_final(&ctx, out);
}

void fbm_sha256d(const void *data, size_t len, uint8_t out[32])
{
    uint8_t first[32];
    fbm_sha256(data, len, first);
    fbm_sha256(first, sizeof first, out);
}
