/* Portable reference SHA-256 (FIPS 180-4). Deliberately straightforward:
 * it is the ground truth that every optimized kernel is tested against. */
#ifndef FBM_SHA256_H
#define FBM_SHA256_H

#include <stddef.h>
#include <stdint.h>

extern const uint32_t fbm_sha256_k[64];
extern const uint32_t fbm_sha256_iv[8];

typedef struct {
    uint32_t state[8];
    uint64_t total;   /* bytes hashed so far */
    uint8_t buf[64];
    size_t buflen;
} fbm_sha256_ctx;

void fbm_sha256_init(fbm_sha256_ctx *ctx);
void fbm_sha256_update(fbm_sha256_ctx *ctx, const void *data, size_t len);
void fbm_sha256_final(fbm_sha256_ctx *ctx, uint8_t out[32]);

void fbm_sha256(const void *data, size_t len, uint8_t out[32]);
/* Bitcoin's double hash: SHA256(SHA256(data)). */
void fbm_sha256d(const void *data, size_t len, uint8_t out[32]);

/* One compression of a 64-byte block into state. */
void fbm_sha256_transform(uint32_t state[8], const uint8_t block[64]);
/* Same, with the 16 message words already loaded (host order). */
void fbm_sha256_transform_words(uint32_t state[8], const uint32_t w[16]);

#endif
