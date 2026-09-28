#include <CL/cl.h>
#include <cstdio>

bool solveDevice(const cl_context cxContext,
                 const cl_command_queue cqCommandQueue,
                 const cl_program cpProgram,
                 const cl_mem clInput,
                 cl_mem clAverage,
                 const int size) {

    cl_int ciErr;

    cl_kernel ckKernel1 = clCreateKernel(cpProgram, "compute", &ciErr);
    if (!ckKernel1 || ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create compute kernel!\n");
        return false;
    }

    ciErr  = clSetKernelArg(ckKernel1, 0, sizeof(cl_mem), &clInput);
    ciErr |= clSetKernelArg(ckKernel1, 1, sizeof(cl_mem), &clAverage);
    ciErr |= clSetKernelArg(ckKernel1, 2, sizeof(int),    &size);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clSetKernelArg.\n", ciErr);
        clReleaseKernel(ckKernel1);
        return false;
    }

    size_t local[1]  = {256};
    size_t global[1] = { (size_t)size };

    ciErr = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel1, 1, NULL,
                                   global, local, 0, NULL, NULL);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clEnqueueNDRangeKernel.\n", ciErr);
        clReleaseKernel(ckKernel1);
        return false;
    }

    clReleaseKernel(ckKernel1);
    return true;
}