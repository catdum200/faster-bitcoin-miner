/* Run-time loading of the OpenCL library (OpenCL.dll, libOpenCL.so.1), so
 * that fbm-gpu needs no OpenCL SDK or import library to build, only a GPU
 * driver to run. Include after <CL/cl.h>: every OpenCL call below is renamed
 * to a function pointer that fbm_cl_load() fills in. */
#ifndef FBM_CLLOAD_H
#define FBM_CLLOAD_H

#define FBM_CL_FUNCS(X)                                                                        \
    X(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id *, cl_uint *))                        \
    X(cl_int, clGetPlatformInfo, (cl_platform_id, cl_platform_info, size_t, void *, size_t *)) \
    X(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id *,        \
                               cl_uint *))                                                     \
    X(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void *, size_t *))       \
    X(cl_context, clCreateContext, (const cl_context_properties *, cl_uint,                    \
                                    const cl_device_id *,                                      \
                                    void(CL_CALLBACK *)(const char *, const void *, size_t,    \
                                                        void *),                               \
                                    void *, cl_int *))                                         \
    X(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id,                       \
                                               cl_command_queue_properties, cl_int *))         \
    X(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char **,              \
                                              const size_t *, cl_int *))                       \
    X(cl_program, clCreateProgramWithBinary, (cl_context, cl_uint, const cl_device_id *,       \
                                              const size_t *, const unsigned char **,          \
                                              cl_int *, cl_int *))                             \
    X(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id *, const char *,        \
                               void(CL_CALLBACK *)(cl_program, void *), void *))               \
    X(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, \
                                      void *, size_t *))                                       \
    X(cl_int, clGetProgramInfo, (cl_program, cl_program_info, size_t, void *, size_t *))       \
    X(cl_kernel, clCreateKernel, (cl_program, const char *, cl_int *))                         \
    X(cl_int, clGetKernelWorkGroupInfo, (cl_kernel, cl_device_id, cl_kernel_work_group_info,   \
                                         size_t, void *, size_t *))                            \
    X(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void *, cl_int *))            \
    X(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void *))                      \
    X(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t *,   \
                                       const size_t *, const size_t *, cl_uint,                \
                                       const cl_event *, cl_event *))                          \
    X(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, \
                                    cl_uint, const cl_event *, cl_event *))                    \
    X(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t,        \
                                     const void *, cl_uint, const cl_event *, cl_event *))     \
    X(cl_int, clFlush, (cl_command_queue))                                                     \
    X(cl_int, clFinish, (cl_command_queue))                                                    \
    X(cl_int, clWaitForEvents, (cl_uint, const cl_event *))                                    \
    X(cl_int, clGetEventProfilingInfo, (cl_event, cl_profiling_info, size_t, void *, size_t *))\
    X(cl_int, clReleaseEvent, (cl_event))                                                      \
    X(cl_int, clReleaseKernel, (cl_kernel))                                                    \
    X(cl_int, clReleaseProgram, (cl_program))                                                  \
    X(cl_int, clReleaseMemObject, (cl_mem))                                                    \
    X(cl_int, clReleaseCommandQueue, (cl_command_queue))                                       \
    X(cl_int, clReleaseContext, (cl_context))

#define FBM_CL_DECLARE(ret, name, params) extern ret(CL_API_CALL *fbm_##name) params;
FBM_CL_FUNCS(FBM_CL_DECLARE)
#undef FBM_CL_DECLARE

#define clGetPlatformIDs fbm_clGetPlatformIDs
#define clGetPlatformInfo fbm_clGetPlatformInfo
#define clGetDeviceIDs fbm_clGetDeviceIDs
#define clGetDeviceInfo fbm_clGetDeviceInfo
#define clCreateContext fbm_clCreateContext
#define clCreateCommandQueue fbm_clCreateCommandQueue
#define clCreateProgramWithSource fbm_clCreateProgramWithSource
#define clCreateProgramWithBinary fbm_clCreateProgramWithBinary
#define clBuildProgram fbm_clBuildProgram
#define clGetProgramBuildInfo fbm_clGetProgramBuildInfo
#define clGetProgramInfo fbm_clGetProgramInfo
#define clCreateKernel fbm_clCreateKernel
#define clGetKernelWorkGroupInfo fbm_clGetKernelWorkGroupInfo
#define clCreateBuffer fbm_clCreateBuffer
#define clSetKernelArg fbm_clSetKernelArg
#define clEnqueueNDRangeKernel fbm_clEnqueueNDRangeKernel
#define clEnqueueReadBuffer fbm_clEnqueueReadBuffer
#define clEnqueueWriteBuffer fbm_clEnqueueWriteBuffer
#define clFlush fbm_clFlush
#define clFinish fbm_clFinish
#define clWaitForEvents fbm_clWaitForEvents
#define clGetEventProfilingInfo fbm_clGetEventProfilingInfo
#define clReleaseEvent fbm_clReleaseEvent
#define clReleaseKernel fbm_clReleaseKernel
#define clReleaseProgram fbm_clReleaseProgram
#define clReleaseMemObject fbm_clReleaseMemObject
#define clReleaseCommandQueue fbm_clReleaseCommandQueue
#define clReleaseContext fbm_clReleaseContext

/* Loads the library and every function; returns 0, or -1 with a message on
 * stderr. Safe to call more than once. */
int fbm_cl_load(void);

#endif
