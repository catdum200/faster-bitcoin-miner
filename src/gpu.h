/* OpenCL driver for the GPU kernels (fbm-gpu). The search contract is the
 * CPU kernels' (kernels.h): hash every (version r, nonce n) with
 * r0 <= r < r0 + nr and n0 <= n < n0 + nn, and report each pair whose final
 * hash has bswap(H7) <= job->t7. */
#ifndef FBM_GPU_H
#define FBM_GPU_H

#include <stdint.h>
#include <stdio.h>

#include "kernels.h"

typedef enum {
    FBM_GPU_NONCE = 0, /* lanes hold nonces (prior-art layout) */
    FBM_GPU_VR = 1     /* lanes hold rolled versions, schedule shared per nonce */
} fbm_gpu_layout;

typedef struct {
    int platform, device;   /* indices from `fbm-gpu list`; -1: first GPU, else first device */
    unsigned wg;            /* work-group size */
    unsigned nonce_iters;   /* nonces per work-item per launch, nonce layout */
    unsigned vr_iters;      /* nonces per work-item per launch, version layout (a wave
                               lives for this many hashes, so keep it small) */
    double launch_ms;       /* target length of one kernel launch. Short launches keep a
                               desktop responsive when the GPU also drives the display */
    double launch_hashes;   /* if > 0, a fixed launch size instead (tests) */
    uint32_t hit_cap;       /* device-side candidate buffer, in (version, nonce) pairs */
    const char *build_opts; /* extra OpenCL compiler options */
    int inject_fault;       /* tests only: corrupt one job word, like a faulty card */
    const char *program;    /* if set, a prebuilt program binary (e.g. the clang-built,
                               audited gfx1200 code object) instead of the source */
} fbm_gpu_opts;

typedef struct fbm_gpu fbm_gpu;

void fbm_gpu_default_opts(fbm_gpu_opts *o);
/* Prints every OpenCL platform and device; returns the number of devices. */
int fbm_gpu_list(void);
/* Opens a device and builds the kernels. Prints errors and returns NULL on
 * failure (no device, build error). */
fbm_gpu *fbm_gpu_open(const fbm_gpu_opts *o);
void fbm_gpu_close(fbm_gpu *g);
const char *fbm_gpu_name(const fbm_gpu *g);
/* Device limits and the compiled kernels' resources (spills, wave size). */
void fbm_gpu_info(fbm_gpu *g, FILE *out);
/* Writes the driver-compiled program binary (an AMDGPU ELF on AMD drivers). */
int fbm_gpu_dump(fbm_gpu *g, const char *path);
/* Returns nr * nn. Candidates go to *out (out->n counts all of them, even
 * beyond out->cap and beyond the device buffer). */
uint64_t fbm_gpu_scan(fbm_gpu *g, fbm_gpu_layout layout, const fbm_job *job, uint32_t r0,
                      uint32_t nr, uint32_t n0, uint64_t nn, fbm_hits *out);
/* Run-time tuning: iters, launch_ms, launch_hashes and hit_cap (up to the
 * capacity allocated at open) may be changed between scans; the rest are
 * fixed. */
fbm_gpu_opts *fbm_gpu_options(fbm_gpu *g);
/* Estimated hash rate of a layout (H/s of kernel time): a model of the device
 * at first, then measured by every scan. Launch sizes follow from it. */
double fbm_gpu_rate(const fbm_gpu *g, fbm_gpu_layout layout);
/* Prior-art baseline: cgminer 3.7.2's poclbm kernel (fetched by
 * bench/gpu-baselines/fetch.sh; not part of fbm). Builds it from `path`;
 * returns 0, or -1 if the file is missing or does not build. */
int fbm_gpu_load_baseline(fbm_gpu *g, const char *path);
/* Scans with the baseline. Unlike fbm_gpu_scan, its nonce loop runs over the
 * big-endian message word W3 = bswap(nonce), so the range [w3_0, w3_0 + nn)
 * is in W3 space; hits are reported as real nonces. It supports only real
 * network targets (t7 = 0) and at most 15 hits per scan (its own design). */
uint64_t fbm_gpu_scan_baseline(fbm_gpu *g, const fbm_job *job, uint32_t r0, uint32_t nr,
                               uint32_t w3_0, uint64_t nn, fbm_hits *out);
/* Runs the issue-rate probes (gpu/probe.cl) and prints a table. Returns 0,
 * or -1 if the probe program does not build on this device. */
int fbm_gpu_probe(fbm_gpu *g, FILE *out);
/* Device (profiling) time of the kernels of the last scan, in seconds. */
double fbm_gpu_kernel_seconds(const fbm_gpu *g);

#endif
