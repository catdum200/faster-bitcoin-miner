/* OpenCL driver for the GPU kernels in gpu/kernels.cl. Portable C11 plus
 * OpenCL 1.2: no threads and no OS-specific calls, so it builds on Linux and
 * on Windows (MSYS2/MinGW) alike. */
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "gpu.h"

#include <stdlib.h>
#include <string.h>

#include "rdna_scalar.h"
#include "gpu_sources.h" /* build/: the OpenCL sources, embedded by tools/embed.c */

/* Must match gpu/kernels.cl. */
#define JOB_WORDS 72
#define JOB_T7 64
#define NNP ((G_RDNA_VR_NN + 15) / 16 * 16)
/* Schedule rows per launch of the version layout (48 words each). */
#define SCHED_ROWS 65536u

#define CL_OK(call)                                                                        \
    do {                                                                                   \
        cl_int e_ = (call);                                                                \
        if (e_ != CL_SUCCESS) {                                                            \
            fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", (int)e_, __FILE__, __LINE__, \
                    #call);                                                                \
            exit(4);                                                                       \
        }                                                                                  \
    } while (0)

struct fbm_gpu {
    fbm_gpu_opts o;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k_nonce, k_sched, k_vr;
    cl_mem job, hits, lanes, sched;
    size_t lanes_words;
    uint32_t hit_alloc; /* pairs allocated; o.hit_cap may be lowered at run time */
    /* The lane table of the last version-layout scan: rebuilding it hashes
     * every version's first block, so it is reused while the job is the same. */
    int lanes_valid;
    uint8_t lanes_hdr[FBM_OFF_NONCE];
    uint32_t lanes_r0, lanes_nr;
    uint32_t *host_hits;
    cl_event *ev;
    size_t nev, cap_ev;
    double ksec;
    double rate[2]; /* H/s of kernel time per layout, for launch sizing */
    int measured[2];
    char name[256];
};

void fbm_gpu_default_opts(fbm_gpu_opts *o)
{
    o->platform = -1;
    o->device = -1;
    o->wg = 64;
    o->nonce_iters = 16;
    o->vr_iters = 64;
    o->launch_ms = 8;
    o->launch_hashes = 0;
    o->hit_cap = 1u << 20;
    o->build_opts = "";
    o->inject_fault = 0;
}

static cl_uint platforms(cl_platform_id *p, cl_uint max)
{
    cl_uint n = 0;
    if (clGetPlatformIDs(max, p, &n) != CL_SUCCESS)
        return 0;
    return n < max ? n : max;
}

static cl_uint devices(cl_platform_id p, cl_device_id *d, cl_uint max)
{
    cl_uint n = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, max, d, &n) != CL_SUCCESS)
        return 0;
    return n < max ? n : max;
}

int fbm_gpu_list(void)
{
    cl_platform_id p[16];
    cl_uint np = platforms(p, 16);
    int total = 0;
    for (cl_uint i = 0; i < np; i++) {
        char pname[256] = "";
        cl_device_id d[16];
        clGetPlatformInfo(p[i], CL_PLATFORM_NAME, sizeof pname, pname, NULL);
        printf("platform %u: %s\n", i, pname);
        cl_uint nd = devices(p[i], d, 16);
        for (cl_uint j = 0; j < nd; j++) {
            char dname[256] = "", ver[256] = "";
            cl_device_type t = 0;
            cl_uint cu = 0, mhz = 0;
            clGetDeviceInfo(d[j], CL_DEVICE_NAME, sizeof dname, dname, NULL);
            clGetDeviceInfo(d[j], CL_DRIVER_VERSION, sizeof ver, ver, NULL);
            clGetDeviceInfo(d[j], CL_DEVICE_TYPE, sizeof t, &t, NULL);
            clGetDeviceInfo(d[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
            clGetDeviceInfo(d[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, NULL);
            printf("  device %u: %s [%s] %u compute units, %u MHz, driver %s\n", j, dname,
                   (t & CL_DEVICE_TYPE_GPU) ? "GPU" : (t & CL_DEVICE_TYPE_CPU) ? "CPU" : "other",
                   cu, mhz, ver);
            total++;
        }
    }
    if (total == 0)
        printf("no OpenCL devices found (is a GPU driver with OpenCL installed?)\n");
    return total;
}

/* Explicit indices, else the first GPU anywhere, else the first device. */
static int pick_device(const fbm_gpu_opts *o, cl_device_id *out)
{
    cl_platform_id p[16];
    cl_uint np = platforms(p, 16);
    cl_device_id first = NULL;
    for (cl_uint i = 0; i < np; i++) {
        cl_device_id d[16];
        cl_uint nd = devices(p[i], d, 16);
        for (cl_uint j = 0; j < nd; j++) {
            cl_device_type t = 0;
            clGetDeviceInfo(d[j], CL_DEVICE_TYPE, sizeof t, &t, NULL);
            if (o->platform >= 0 || o->device >= 0) {
                if ((int)i == (o->platform < 0 ? 0 : o->platform) &&
                    (int)j == (o->device < 0 ? 0 : o->device)) {
                    *out = d[j];
                    return 0;
                }
                continue;
            }
            if (t & CL_DEVICE_TYPE_GPU) {
                *out = d[j];
                return 0;
            }
            if (!first)
                first = d[j];
        }
    }
    if (first && o->platform < 0 && o->device < 0) {
        *out = first;
        return 0;
    }
    return -1;
}

fbm_gpu *fbm_gpu_open(const fbm_gpu_opts *o)
{
    fbm_gpu *g = calloc(1, sizeof *g);
    cl_int err;
    char opts[1024];

    g->o = *o;
    if (pick_device(o, &g->dev) != 0) {
        fprintf(stderr, "no matching OpenCL device (see `fbm-gpu list`)\n");
        free(g);
        return NULL;
    }
    clGetDeviceInfo(g->dev, CL_DEVICE_NAME, sizeof g->name, g->name, NULL);
    /* Until the first scan measures it, model the rate as 64 lanes per
     * compute unit at the maximum clock and ~2600 instructions per hash (an
     * RX 9060 XT comes out at ~2.5 GH/s). */
    cl_uint cu = 1, mhz = 1000;
    clGetDeviceInfo(g->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    clGetDeviceInfo(g->dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, NULL);
    g->rate[0] = g->rate[1] = 64.0 * cu * mhz * 1e6 / 2600;
    g->ctx = clCreateContext(NULL, 1, &g->dev, NULL, NULL, &err);
    CL_OK(err);
    g->q = clCreateCommandQueue(g->ctx, g->dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_OK(err);

    g->prog = clCreateProgramWithSource(g->ctx, fbm_gpu_src_COUNT, (const char **)fbm_gpu_src,
                                        NULL, &err);
    CL_OK(err);
    snprintf(opts, sizeof opts, "-cl-std=CL1.2 -DFBM_WG=%u %s", o->wg, o->build_opts);
    err = clBuildProgram(g->prog, 1, &g->dev, opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(g->prog, g->dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
        char *log = malloc(n + 1);
        clGetProgramBuildInfo(g->prog, g->dev, CL_PROGRAM_BUILD_LOG, n, log, NULL);
        log[n] = '\0';
        fprintf(stderr, "OpenCL build failed (%d) for %s:\n%s\n", (int)err, g->name, log);
        free(log);
        fbm_gpu_close(g);
        return NULL;
    }
    g->k_nonce = clCreateKernel(g->prog, "fbm_nonce", &err);
    CL_OK(err);
    g->k_sched = clCreateKernel(g->prog, "fbm_vr_sched", &err);
    CL_OK(err);
    g->k_vr = clCreateKernel(g->prog, "fbm_vr", &err);
    CL_OK(err);

    g->job = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY, JOB_WORDS * 4, NULL, &err);
    CL_OK(err);
    g->hits = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, (2 + 2 * (size_t)o->hit_cap) * 4, NULL,
                             &err);
    CL_OK(err);
    g->sched = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, (size_t)SCHED_ROWS * NNP * 4, NULL, &err);
    CL_OK(err);
    g->host_hits = malloc((2 + 2 * (size_t)o->hit_cap) * 4);
    g->hit_alloc = o->hit_cap;
    return g;
}

void fbm_gpu_close(fbm_gpu *g)
{
    if (!g)
        return;
    if (g->lanes)
        clReleaseMemObject(g->lanes);
    if (g->sched)
        clReleaseMemObject(g->sched);
    if (g->hits)
        clReleaseMemObject(g->hits);
    if (g->job)
        clReleaseMemObject(g->job);
    if (g->k_vr)
        clReleaseKernel(g->k_vr);
    if (g->k_sched)
        clReleaseKernel(g->k_sched);
    if (g->k_nonce)
        clReleaseKernel(g->k_nonce);
    if (g->prog)
        clReleaseProgram(g->prog);
    if (g->q)
        clReleaseCommandQueue(g->q);
    if (g->ctx)
        clReleaseContext(g->ctx);
    free(g->host_hits);
    free(g->ev);
    free(g);
}

const char *fbm_gpu_name(const fbm_gpu *g)
{
    return g->name;
}

fbm_gpu_opts *fbm_gpu_options(fbm_gpu *g)
{
    return &g->o;
}

double fbm_gpu_rate(const fbm_gpu *g, fbm_gpu_layout layout)
{
    return g->rate[layout];
}

/* Hashes per launch: fixed (tests) or launch_ms at the estimated rate. */
static double launch_size(const fbm_gpu *g, fbm_gpu_layout layout)
{
    if (g->o.launch_hashes > 0)
        return g->o.launch_hashes;
    return g->rate[layout] * g->o.launch_ms * 1e-3;
}

double fbm_gpu_kernel_seconds(const fbm_gpu *g)
{
    return g->ksec;
}

static void kernel_info(fbm_gpu *g, cl_kernel k, const char *name, FILE *out)
{
    size_t wg = 0, mult = 0;
    cl_ulong priv = 0, local = 0;
    clGetKernelWorkGroupInfo(k, g->dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof wg, &wg, NULL);
    clGetKernelWorkGroupInfo(k, g->dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof mult,
                             &mult, NULL);
    clGetKernelWorkGroupInfo(k, g->dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof priv, &priv, NULL);
    clGetKernelWorkGroupInfo(k, g->dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof local, &local, NULL);
    fprintf(out, "  %-13s max work-group %zu, wave/warp size (preferred multiple) %zu, "
            "private (spill) bytes %llu, local bytes %llu\n",
            name, wg, mult, (unsigned long long)priv, (unsigned long long)local);
}

void fbm_gpu_info(fbm_gpu *g, FILE *out)
{
    char s[256] = "";
    cl_uint cu = 0, mhz = 0;
    cl_ulong mem = 0;
    fprintf(out, "device: %s\n", g->name);
    clGetDeviceInfo(g->dev, CL_DEVICE_VENDOR, sizeof s, s, NULL);
    fprintf(out, "  vendor: %s\n", s);
    clGetDeviceInfo(g->dev, CL_DEVICE_VERSION, sizeof s, s, NULL);
    fprintf(out, "  version: %s\n", s);
    clGetDeviceInfo(g->dev, CL_DRIVER_VERSION, sizeof s, s, NULL);
    fprintf(out, "  driver: %s\n", s);
    clGetDeviceInfo(g->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    clGetDeviceInfo(g->dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, NULL);
    clGetDeviceInfo(g->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, NULL);
    fprintf(out, "  %u compute units, max clock %u MHz, %.1f GiB\n", cu, mhz,
            mem / 1073741824.0);
    fprintf(out, "kernels (work-group size %u):\n", g->o.wg);
    kernel_info(g, g->k_nonce, "fbm_nonce", out);
    kernel_info(g, g->k_sched, "fbm_vr_sched", out);
    kernel_info(g, g->k_vr, "fbm_vr", out);
}

int fbm_gpu_dump(fbm_gpu *g, const char *path)
{
    size_t size = 0;
    CL_OK(clGetProgramInfo(g->prog, CL_PROGRAM_BINARY_SIZES, sizeof size, &size, NULL));
    unsigned char *bin = malloc(size ? size : 1);
    unsigned char *bins[1] = {bin};
    CL_OK(clGetProgramInfo(g->prog, CL_PROGRAM_BINARIES, sizeof bins, bins, NULL));
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(bin, 1, size, f) != size) {
        perror(path);
        free(bin);
        if (f)
            fclose(f);
        return -1;
    }
    fclose(f);
    free(bin);
    printf("wrote %zu bytes of compiled program to %s\n", size, path);
    return 0;
}

static cl_event *next_event(fbm_gpu *g)
{
    if (g->nev == g->cap_ev) {
        g->cap_ev = g->cap_ev ? 2 * g->cap_ev : 256;
        g->ev = realloc(g->ev, g->cap_ev * sizeof *g->ev);
    }
    return &g->ev[g->nev++];
}

static void launch(fbm_gpu *g, cl_kernel k, cl_uint dims, const size_t *global,
                   const size_t *local)
{
    CL_OK(clEnqueueNDRangeKernel(g->q, k, dims, NULL, global, local, 0, NULL, next_event(g)));
    /* Keep the queue from growing without bound on long scans. */
    if (g->nev % 64 == 0)
        CL_OK(clFlush(g->q));
}

static void set_u32(cl_kernel k, cl_uint i, uint32_t v)
{
    CL_OK(clSetKernelArg(k, i, sizeof(cl_uint), &v));
}

static void set_mem(cl_kernel k, cl_uint i, cl_mem m)
{
    CL_OK(clSetKernelArg(k, i, sizeof(cl_mem), &m));
}

static void write_job(fbm_gpu *g, const uint32_t *J, size_t nj, uint32_t t7)
{
    uint32_t w[JOB_WORDS] = {0};
    memcpy(w, J, nj * 4);
    w[JOB_T7] = t7;
    if (g->o.inject_fault)
        w[0] ^= 1u << 7;
    CL_OK(clEnqueueWriteBuffer(g->q, g->job, CL_TRUE, 0, sizeof w, w, 0, NULL, NULL));
}

static void reset_hits(fbm_gpu *g)
{
    if (g->o.hit_cap > g->hit_alloc)
        g->o.hit_cap = g->hit_alloc;
    const uint32_t h[2] = {0, g->o.hit_cap};
    CL_OK(clEnqueueWriteBuffer(g->q, g->hits, CL_TRUE, 0, sizeof h, h, 0, NULL, NULL));
}

/* Waits for the queue, sums kernel time, and reads the hit list. Each hit is
 * (index, nonce); map(index) gives the version to report. Scans of at least
 * `hashes` update the layout's rate estimate. */
static void collect(fbm_gpu *g, fbm_gpu_layout layout, double hashes, fbm_hits *out,
                    uint32_t base, uint32_t r_off)
{
    CL_OK(clFinish(g->q));
    g->ksec = 0;
    for (size_t i = 0; i < g->nev; i++) {
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(g->ev[i], CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
        clGetEventProfilingInfo(g->ev[i], CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
        g->ksec += (double)(t1 - t0) * 1e-9;
        clReleaseEvent(g->ev[i]);
    }
    g->nev = 0;
    /* Tiny scans are dominated by launch overhead; do not learn from them. */
    if (g->ksec > 2e-3 && hashes > 0) {
        const double r = hashes / g->ksec;
        g->rate[layout] = g->measured[layout] ? 0.5 * g->rate[layout] + 0.5 * r : r;
        g->measured[layout] = 1;
    }

    uint32_t *h = g->host_hits;
    CL_OK(clEnqueueReadBuffer(g->q, g->hits, CL_TRUE, 0, 8, h, 0, NULL, NULL));
    const uint32_t n = h[0], stored = n < g->o.hit_cap ? n : g->o.hit_cap;
    if (stored)
        CL_OK(clEnqueueReadBuffer(g->q, g->hits, CL_TRUE, 8, (size_t)stored * 8, h + 2, 0, NULL,
                                  NULL));
    for (uint32_t i = 0; i < stored; i++)
        fbm_hits_push(out, fbm_rolled_version(base, r_off + h[2 + 2 * i]), h[3 + 2 * i]);
    /* Candidates the device buffer could not hold still count. */
    out->n += n - stored;
}

static uint64_t scan_nonce(fbm_gpu *g, const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                           uint64_t nn, fbm_hits *out)
{
    const uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);
    const uint64_t per_group = (uint64_t)g->o.wg * g->o.nonce_iters;
    uint64_t per_launch = (uint64_t)(launch_size(g, FBM_GPU_NONCE) / per_group) * per_group;

    if (per_launch < per_group)
        per_launch = per_group;
    if (per_launch > (1ull << 31))
        per_launch = 1ull << 31;
    reset_hits(g);
    for (uint32_t r = r0; r < r0 + nr; r++) {
        uint32_t in[11], J[G_RDNA_NONCE_NJ];
        fbm_job_inputs(job->header, fbm_rolled_version(base, r), in);
        g_rdna_nonce_job(in, J);
        write_job(g, J, G_RDNA_NONCE_NJ, job->t7);
        for (uint64_t off = 0; off < nn; off += per_launch) {
            const uint64_t cnt = nn - off < per_launch ? nn - off : per_launch;
            const size_t global = (size_t)((cnt + per_group - 1) / per_group) * g->o.wg;
            const size_t local = g->o.wg;
            set_mem(g->k_nonce, 0, g->job);
            set_u32(g->k_nonce, 1, (uint32_t)(n0 + off));
            set_u32(g->k_nonce, 2, (uint32_t)cnt);
            set_u32(g->k_nonce, 3, g->o.nonce_iters);
            set_u32(g->k_nonce, 4, r - r0);
            set_mem(g->k_nonce, 5, g->hits);
            launch(g, g->k_nonce, 1, &global, &local);
        }
    }
    collect(g, FBM_GPU_NONCE, (double)nr * nn, out, base, r0);
    return (uint64_t)nr * nn;
}

static uint64_t scan_vr(fbm_gpu *g, const fbm_job *job, uint32_t r0, uint32_t nr, uint32_t n0,
                        uint64_t nn, fbm_hits *out)
{
    const uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION);
    const uint32_t wg = g->o.wg;
    const size_t nrp = ((size_t)nr + wg - 1) / wg * wg; /* versions padded to whole groups */
    uint32_t in[11], J[G_RDNA_VR_NJ];
    cl_int err;

    /* JOB values depend only on W0..W2, not on the version. */
    fbm_job_inputs(job->header, base, in);
    g_rdna_vr_job(in, J);
    write_job(g, J, G_RDNA_VR_NJ, job->t7);

    /* LANE values: midstate and rounds 0-2 of each version, slot-major so
     * that a wave's loads are contiguous. Padding lanes repeat version r0. */
    if (!g->lanes_valid || g->lanes_r0 != r0 || g->lanes_nr != nr ||
        memcmp(g->lanes_hdr, job->header, FBM_OFF_NONCE) != 0) {
        if (g->lanes_words < nrp * G_RDNA_VR_NL) {
            if (g->lanes)
                clReleaseMemObject(g->lanes);
            g->lanes_words = nrp * G_RDNA_VR_NL;
            g->lanes = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY, g->lanes_words * 4, NULL, &err);
            CL_OK(err);
        }
        uint32_t *tab = malloc(nrp * G_RDNA_VR_NL * 4);
        for (size_t i = 0; i < nrp; i++) {
            uint32_t lin[11], Ls[G_RDNA_VR_NL];
            fbm_job_inputs(job->header,
                           fbm_rolled_version(base, r0 + (i < nr ? (uint32_t)i : 0)), lin);
            g_rdna_vr_lane(lin, J, Ls);
            for (int k = 0; k < G_RDNA_VR_NL; k++)
                tab[k * nrp + i] = Ls[k];
        }
        CL_OK(clEnqueueWriteBuffer(g->q, g->lanes, CL_TRUE, 0, nrp * G_RDNA_VR_NL * 4, tab, 0,
                                   NULL, NULL));
        free(tab);
        memcpy(g->lanes_hdr, job->header, FBM_OFF_NONCE);
        g->lanes_r0 = r0;
        g->lanes_nr = nr;
        g->lanes_valid = 1;
    }
    reset_hits(g);

    uint64_t per_launch = (uint64_t)(launch_size(g, FBM_GPU_VR) / (double)nrp);
    if (per_launch < 1)
        per_launch = 1;
    if (per_launch > SCHED_ROWS)
        per_launch = SCHED_ROWS;
    for (uint64_t off = 0; off < nn; off += per_launch) {
        const uint32_t cnt = (uint32_t)(nn - off < per_launch ? nn - off : per_launch);
        const uint32_t iters = cnt < g->o.vr_iters ? cnt : g->o.vr_iters;
        const size_t sglobal = ((size_t)cnt + wg - 1) / wg * wg, slocal = wg;
        set_mem(g->k_sched, 0, g->job);
        set_u32(g->k_sched, 1, (uint32_t)(n0 + off));
        set_u32(g->k_sched, 2, cnt);
        set_mem(g->k_sched, 3, g->sched);
        launch(g, g->k_sched, 1, &sglobal, &slocal);

        const size_t global[2] = {nrp, (cnt + iters - 1) / iters}, local[2] = {wg, 1};
        set_mem(g->k_vr, 0, g->job);
        set_mem(g->k_vr, 1, g->lanes);
        set_u32(g->k_vr, 2, (uint32_t)nrp);
        set_u32(g->k_vr, 3, nr);
        set_mem(g->k_vr, 4, g->sched);
        set_u32(g->k_vr, 5, (uint32_t)(n0 + off));
        set_u32(g->k_vr, 6, cnt);
        set_u32(g->k_vr, 7, iters);
        set_mem(g->k_vr, 8, g->hits);
        launch(g, g->k_vr, 2, global, local);
    }
    collect(g, FBM_GPU_VR, (double)nrp * nn, out, base, r0);
    return (uint64_t)nr * nn;
}

uint64_t fbm_gpu_scan(fbm_gpu *g, fbm_gpu_layout layout, const fbm_job *job, uint32_t r0,
                      uint32_t nr, uint32_t n0, uint64_t nn, fbm_hits *out)
{
    if (nr == 0 || nn == 0)
        return 0;
    return layout == FBM_GPU_VR ? scan_vr(g, job, r0, nr, n0, nn, out)
                                : scan_nonce(g, job, r0, nr, n0, nn, out);
}
