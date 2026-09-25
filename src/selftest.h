#ifndef FBM_SELFTEST_H
#define FBM_SELFTEST_H

extern const char *const fbm_genesis_hex;
extern const char *const fbm_block125552_hex;

/* Returns 0 when every test passes. */
int fbm_selftest(void);

#endif
