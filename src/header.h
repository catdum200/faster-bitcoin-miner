/* Bitcoin block-header helpers: hex, field access, compact-bits targets. */
#ifndef FBM_HEADER_H
#define FBM_HEADER_H

#include <stddef.h>
#include <stdint.h>

#define FBM_HEADER_LEN 80
#define FBM_OFF_VERSION 0
#define FBM_OFF_PREV 4
#define FBM_OFF_MERKLE 36
#define FBM_OFF_TIME 68
#define FBM_OFF_BITS 72
#define FBM_OFF_NONCE 76

/* Returns 0 on success, -1 on bad length or characters. */
int fbm_hex_decode(const char *hex, uint8_t *out, size_t out_len);
void fbm_hex_encode(const uint8_t *in, size_t len, char *out /* 2*len+1 */);
/* Block hashes are conventionally displayed byte-reversed. */
void fbm_hash_display(const uint8_t hash[32], char out[65]);

uint32_t fbm_header_get(const uint8_t *hdr, size_t off);
void fbm_header_set(uint8_t *hdr, size_t off, uint32_t v);

/* Expand compact "bits" into a 256-bit little-endian target.
 * Returns 0 on success, -1 for negative or overflowing encodings. */
int fbm_bits_to_target(uint32_t bits, uint8_t target_le[32]);
/* Compare two 256-bit little-endian numbers: <0, 0, >0 like memcmp. */
int fbm_cmp256_le(const uint8_t a[32], const uint8_t b[32]);
/* Top 32 bits of a 256-bit little-endian number (bytes 28..31). */
uint32_t fbm_top32_le(const uint8_t v[32]);
/* Bitcoin difficulty of a target (difficulty 1 = bits 0x1d00ffff). */
double fbm_target_difficulty(const uint8_t target_le[32]);

#endif
