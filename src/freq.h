#ifndef FBM_FREQ_H
#define FBM_FREQ_H

/* Measured core clock in GHz while running scalar, 256-bit and 512-bit
 * integer code. Call only the probes the CPU supports. */
double fbm_ghz_scalar(void);
double fbm_ghz_ymm(void);
double fbm_ghz_zmm(void);

#endif
