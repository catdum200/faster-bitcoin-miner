/* Real mainnet block headers with known (version, nonce) answers. */
#ifndef FBM_BLOCKS_H
#define FBM_BLOCKS_H

#include <stddef.h>

typedef struct {
    const char *name; /* height */
    const char *hex;  /* 80-byte header, 160 hex characters */
} fbm_known_block;

extern const char *const fbm_genesis_hex;
extern const char *const fbm_block125552_hex;
/* Genesis, 125552, then 7 blocks whose versions have BIP 320 bits rolled. */
extern const fbm_known_block fbm_known_blocks[];
extern const size_t fbm_known_block_count;

#endif
