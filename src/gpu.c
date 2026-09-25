/* OpenCL driver for the GPU kernels in gpu/kernels.cl. Portable C11 plus
 * OpenCL 1.2: no threads and no OS-specific calls, so it builds on Linux and
 * on Windows (MSYS2/MinGW) alike. */
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "clload.h"
#include "gpu.h"

#include <stdlib.h>
#include <string.h>

#include "rdna_scalar.h"
#include "gpu_probe_source.h"
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
    cl_program base_prog; /* optional poclbm baseline */
    cl_kernel k_base;
    cl_mem base_out;
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
    o->program = NULL;
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
    if (fbm_cl_load() != 0)
        return 0;
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
    if (fbm_cl_load() != 0)
        return NULL;
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

    if (o->program) {
        /* e.g. `make gpu-codeobj`: then the ISA that runs is the audited one */
        FILE *f = fopen(o->program, "rb");
        if (!f) {
            perror(o->program);
            fbm_gpu_close(g);
            return NULL;
        }
        fseek(f, 0, SEEK_END);
        const size_t n = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        unsigned char *bin = malloc(n ? n : 1);
        const size_t got = fread(bin, 1, n, f);
        fclose(f);
        const unsigned char *bins[1] = {bin};
        cl_int status = CL_SUCCESS;
        g->prog = clCreateProgramWithBinary(g->ctx, 1, &g->dev, &got, bins, &status, &err);
        free(bin);
        if (err != CL_SUCCESS || status != CL_SUCCESS) {
            fprintf(stderr, "the runtime rejects %s (error %d/%d); run from source instead\n",
                    o->program, (int)err, (int)status);
            g->prog = NULL;
            fbm_gpu_close(g);
            return NULL;
        }
    } else {
        g->prog = clCreateProgramWithSource(g->ctx, fbm_gpu_src_COUNT,
                                            (const char **)fbm_gpu_src, NULL, &err);
        CL_OK(err);
    }
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
    if (g->k_base)
        clReleaseKernel(g->k_base);
    if (g->base_prog)
        clReleaseProgram(g->base_prog);
    if (g->base_out)
        clReleaseMemObject(g->base_out);
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
static void collect_time(fbm_gpu *g, fbm_gpu_layout layout, double hashes)
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
}

static void collect(fbm_gpu *g, fbm_gpu_layout layout, double hashes, fbm_hits *out,
                    uint32_t base, uint32_t r_off)
{
    collect_time(g, layout, hashes);

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

/* ---- prior-art baseline: cgminer 3.7.2 poclbm --------------------------- */

int fbm_gpu_load_baseline(fbm_gpu *g, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = malloc((size_t)n + 1);
    const size_t got = fread(src, 1, (size_t)n, f);
    fclose(f);
    src[got] = '\0';
    cl_int err;
    const char *srcs[1] = {src};
    g->base_prog = clCreateProgramWithSource(g->ctx, 1, srcs, NULL, &err);
    free(src);
    CL_OK(err);
    /* As cgminer builds it for GCN: amd_bitalign where the driver has AMD's
     * media ops, no BFI_INT binary patching, one nonce per work-item. */
    char ext[4096] = "", opts[256];
    clGetDeviceInfo(g->dev, CL_DEVICE_EXTENSIONS, sizeof ext, ext, NULL);
    snprintf(opts, sizeof opts, "-DWORKSIZE=%u%s", g->o.wg,
             strstr(ext, "cl_amd_media_ops") ? " -DBITALIGN" : "");
    if (clBuildProgram(g->base_prog, 1, &g->dev, opts, NULL, NULL) != CL_SUCCESS) {
        clReleaseProgram(g->base_prog);
        g->base_prog = NULL;
        return -1;
    }
    g->k_base = clCreateKernel(g->base_prog, "search", &err);
    CL_OK(err);
    g->base_out = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, 16 * 4, NULL, &err);
    CL_OK(err);
    return 0;
}

/* The kernel's 26 per-job arguments (all but the nonce base and the output
 * buffer), derived clean-room from how the kernel consumes them, not from
 * cgminer's host code. With (A..H) the state after rounds 0-2 of block 2 and
 * T = H + S1(E) + Ch(E,F,G) + K3, round 3 gives e = D + T + W3 and
 * a = T + S0(A) + Maj(A,B,C) + W3; the rest are round-4..6 partial sums and
 * the constant parts of schedule words W16..W19, W23, W24 and W31..W33. */
static void poclbm_args(const uint32_t in[11], uint32_t a[26])
{
    const uint32_t *K = fbm_sha256_k;
    uint32_t s[8];
    memcpy(s, in, sizeof s);
    for (int t = 0; t < 3; t++) {
        const uint32_t t1 = s[7] + S_BSIG1(s[4]) + S_CH(s[4], s[5], s[6]) + K[t] + in[8 + t];
        const uint32_t t2 = S_BSIG0(s[0]) + S_MAJ(s[0], s[1], s[2]);
        memmove(s + 1, s, 7 * sizeof s[0]);
        s[4] += t1;
        s[0] = t1 + t2;
    }
    const uint32_t A = s[0], B = s[1], C = s[2], D = s[3], E = s[4], F = s[5], G = s[6], H = s[7];
    const uint32_t W0 = in[8], W1 = in[9], W2 = in[10];
    const uint32_t W16 = S_SSIG0(W1) + W0, W17 = S_SSIG1(0x280u) + S_SSIG0(W2) + W1;
    const uint32_t T = H + S_BSIG1(E) + S_CH(E, F, G) + K[3];
    memcpy(a, in, 8 * sizeof a[0]); /* state0..7: the midstate */
    a[8] = E;                         /* b1 */
    a[9] = F;                         /* c1 */
    a[10] = A;                        /* f1 */
    a[11] = B;                        /* g1 */
    a[12] = C;                        /* h1 */
    a[13] = W16;                      /* fw0 */
    a[14] = W17;                      /* fw1 */
    a[15] = S_SSIG1(W16) + W2;        /* fw2 */
    a[16] = S_SSIG1(W17) + S_SSIG0(0x80000000u); /* fw3 */
    a[17] = S_SSIG0(W16) + 0x280u;    /* fw15 */
    a[18] = S_SSIG0(W17) + W16;       /* fw01r */
    a[19] = G + K[4] + 0x80000000u;   /* D1A */
    a[20] = F + K[5];                 /* C1addK5 */
    a[21] = E + K[6];                 /* B1addK6 */
    a[22] = W16 + K[16];              /* W16addK16 */
    a[23] = W17 + K[17];              /* W17addK17 */
    a[24] = T + S_BSIG0(A) + S_MAJ(A, B, C); /* PreVal4addT1 */
    a[25] = T + D;                    /* Preval0 */
}

uint64_t fbm_gpu_scan_baseline(fbm_gpu *g, const fbm_job *job, uint32_t r0, uint32_t nr,
                               uint32_t w3_0, uint64_t nn, fbm_hits *out)
{
    const uint32_t base = fbm_header_get(job->header, FBM_OFF_VERSION), wg = g->o.wg;
    uint64_t per_launch = (uint64_t)(launch_size(g, FBM_GPU_NONCE) / wg) * wg;
    per_launch = per_launch < wg ? wg : per_launch > (1ull << 31) ? 1ull << 31 : per_launch;
    for (uint32_t r = r0; r < r0 + nr; r++) {
        const uint32_t version = fbm_rolled_version(base, r);
        uint32_t in[11], a[26], zero[16] = {0};
        fbm_job_inputs(job->header, version, in);
        poclbm_args(in, a);
        CL_OK(clEnqueueWriteBuffer(g->q, g->base_out, CL_TRUE, 0, sizeof zero, zero, 0, NULL,
                                   NULL));
        for (cl_uint i = 0; i < 13; i++)
            set_u32(g->k_base, i, a[i]);
        for (cl_uint i = 13; i < 26; i++)
            set_u32(g->k_base, i + 1, a[i]); /* argument 13 is the nonce base */
        set_mem(g->k_base, 27, g->base_out);
        for (uint64_t off = 0; off < nn; off += per_launch) {
            const uint64_t cnt = nn - off < per_launch ? nn - off : per_launch;
            const size_t global = (size_t)((cnt + wg - 1) / wg) * wg, local = wg;
            set_u32(g->k_base, 13, (uint32_t)(w3_0 + off));
            launch(g, g->k_base, 1, &global, &local);
        }
        uint32_t o[16];
        CL_OK(clEnqueueReadBuffer(g->q, g->base_out, CL_TRUE, 0, sizeof o, o, 0, NULL, NULL));
        const uint32_t n = o[15] < 15 ? o[15] : 15;
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t w3 = o[i];
            if ((uint32_t)(w3 - w3_0) < nn) /* the last work-group runs past the range */
                fbm_hits_push(out, version, fbm_bswap32(w3));
        }
        if (o[15] > 15)
            out->n += o[15] - 15; /* its buffer holds 15; the rest are lost */
    }
    collect_time(g, FBM_GPU_NONCE, (double)nr * nn);
    return (uint64_t)nr * nn;
}

/* ---- issue-rate probes ---------------------------------------------------- */

static double run_probe(fbm_gpu *g, cl_kernel k, size_t global, uint32_t iters)
{
    const size_t local = 64;
    cl_event ev;
    cl_ulong t0 = 0, t1 = 0;
    set_u32(k, 0, iters);
    set_u32(k, 1, 12345);
    set_mem(k, 2, g->job);
    CL_OK(clEnqueueNDRangeKernel(g->q, k, 1, NULL, &global, &local, 0, NULL, &ev));
    CL_OK(clWaitForEvents(1, &ev));
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
    clReleaseEvent(ev);
    return (double)(t1 - t0) * 1e-9;
}

int fbm_gpu_probe(fbm_gpu *g, FILE *out)
{
    static const char *const names[] = {"probe_add", "probe_alignbit", "probe_xor3",
                                        "probe_add3", "probe_bfi", "probe_xad",
                                        "probe_mix_salu", "probe_mix_delay"};
    enum { N = sizeof names / sizeof names[0] };
    cl_int err;
    cl_uint cu = 1;
    clGetDeviceInfo(g->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    cl_program p = clCreateProgramWithSource(g->ctx, fbm_gpu_probe_src_COUNT,
                                             (const char **)fbm_gpu_probe_src, NULL, &err);
    CL_OK(err);
    if (clBuildProgram(p, 1, &g->dev, "-cl-std=CL1.2", NULL, NULL) != CL_SUCCESS) {
        fprintf(out, "the probe program does not build on this device (inline asm?)\n");
        clReleaseProgram(p);
        return -1;
    }
    /* 256 work-groups of 64 per compute unit, so every SIMD has many waves
     * to pick from, as in the mining kernels. */
    const size_t global = (size_t)cu * 64 * 256;
    double secs[N], ops[N];
    for (int i = 0; i < N; i++) {
        cl_kernel k = clCreateKernel(p, names[i], &err);
        CL_OK(err);
        uint32_t iters = 4;
        double t = run_probe(g, k, global, iters); /* warm-up and calibration */
        while (t < 0.05 && iters < (1u << 24)) {
            iters *= 4;
            t = run_probe(g, k, global, iters);
        }
        double best = t;
        for (int r = 0; r < 4; r++) {
            t = run_probe(g, k, global, iters);
            best = t < best ? t : best;
        }
        secs[i] = best;
        ops[i] = (double)global * iters * 64; /* VALU lane-ops (the mixes add non-VALU) */
        clReleaseKernel(k);
    }
    clReleaseProgram(p);
    const double add = ops[0] / secs[0];
    fprintf(out, "| probe (64 instructions per iteration) | VALU lane-ops/s | rate vs v_add |\n"
            "|---|---:|---:|\n");
    for (int i = 0; i < N; i++)
        fprintf(out, "| %s | %.3g | %.3f |\n", names[i], ops[i] / secs[i], ops[i] / secs[i] / add);
    fprintf(out, "\nv_add_nc_u32 runs at %.3g lane-ops/s. If it is full rate (32 lanes per SIMD\n"
            "per clock), the shader clock is %.2f GHz with 64 lanes per reported compute unit\n"
            "(%u reported), or %.2f GHz if the driver reports dual-CU WGPs. For an RX 9060 XT\n"
            "(2048 lanes) that is %.2f GHz.\n", add, add / (64.0 * cu) * 1e-9, cu,
            add / (128.0 * cu) * 1e-9, add / 2048 * 1e-9);
    fprintf(out, "probe_mix_salu adds 64 s_xor_b32 and probe_mix_delay 64 s_delay_alu to the 64\n"
            "v_add of each iteration. A rate near 1.0 means those instructions are free (they\n"
            "take no vector issue slot); near 0.5 means each costs one. This decides whether\n"
            "the vr kernel's gain is nearer 1.17x or 1.10x.\n");
    return 0;
}

uint64_t fbm_gpu_scan(fbm_gpu *g, fbm_gpu_layout layout, const fbm_job *job, uint32_t r0,
                      uint32_t nr, uint32_t n0, uint64_t nn, fbm_hits *out)
{
    if (nr == 0 || nn == 0)
        return 0;
    return layout == FBM_GPU_VR ? scan_vr(g, job, r0, nr, n0, nn, out)
                                : scan_nonce(g, job, r0, nr, n0, nn, out);
}
