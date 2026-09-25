/* Run-time loading of the OpenCL library; see clload.h. */
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "clload.h"

#define FBM_CL_DEFINE(ret, name, params) ret(CL_API_CALL *fbm_##name) params;
FBM_CL_FUNCS(FBM_CL_DEFINE)
#undef FBM_CL_DEFINE

int fbm_cl_load(void)
{
    static int state; /* 0 not tried, 1 loaded, -1 failed */
    if (state)
        return state > 0 ? 0 : -1;
    state = -1;
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("OpenCL.dll");
#define FBM_SYM(name) ((void *)GetProcAddress(lib, name))
#else
    static const char *const names[] = {
#ifdef __APPLE__
        "/System/Library/Frameworks/OpenCL.framework/OpenCL",
#endif
        "libOpenCL.so.1", "libOpenCL.so"};
    void *lib = NULL;
    for (size_t i = 0; !lib && i < sizeof names / sizeof names[0]; i++)
        lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
#define FBM_SYM(name) dlsym(lib, name)
#endif
    if (!lib) {
        fprintf(stderr, "no OpenCL runtime found. Install your GPU driver: AMD's Adrenalin "
                "driver (Windows) includes OpenCL; on Linux install ROCm's OpenCL runtime or "
                "Mesa's rusticl, plus an ICD loader (ocl-icd-libopencl1).\n");
        return -1;
    }
    /* ISO C has no cast from void * to a function pointer; a union does it on
     * every compiler this builds with. */
#define FBM_CL_LOAD(ret, name, params)                                                    \
    {                                                                                     \
        union {                                                                           \
            void *p;                                                                      \
            ret(CL_API_CALL *f) params;                                                   \
        } u;                                                                              \
        u.p = FBM_SYM(#name);                                                             \
        if (!u.p) {                                                                       \
            fprintf(stderr, "the OpenCL runtime lacks %s (OpenCL 1.2 is required)\n", #name); \
            return -1;                                                                    \
        }                                                                                 \
        fbm_##name = u.f;                                                                 \
    }
    FBM_CL_FUNCS(FBM_CL_LOAD)
#undef FBM_CL_LOAD
    state = 1;
    return 0;
}
