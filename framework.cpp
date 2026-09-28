#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#define R 128
#define SIZE (1024*1024*32)

#include "kernel_C.cpp"
#include "kernel_exec.cpp"

void startTimer(timeval *timer) {
    gettimeofday(timer, NULL);
}

int stopTimer(timeval *timer) {
    timeval tmp;
    gettimeofday(&tmp, NULL);
    tmp.tv_sec  -= timer->tv_sec;
    tmp.tv_usec -= timer->tv_usec;
    if (tmp.tv_usec < 0) {
        tmp.tv_usec += 1000000;
        tmp.tv_sec--;
    }
    return int(tmp.tv_usec + tmp.tv_sec*1000000);
}

static char *load_program_source(const char *filename) {
    struct stat statbuf;
    FILE   *fh;
    char   *source;
    size_t fsize;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    fsize = statbuf.st_size;
    source = (char *) malloc(fsize + 1);
    fread(source, fsize, 1, fh);
    source[fsize] = '\0';
    fclose(fh);

    return source;
}

void printDeviceInfo(cl_device_id device_id) {
    char device_string[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf(" CL_DEVICE_NAME: %s\n", device_string);
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    printf(" CL_DEVICE_VENDOR: %s\n", device_string);
}

int main(int argc, char **argv) {
    float *input       = NULL;  // input array
    float *average     = NULL;  // CPU reference output
    float *averageTest = NULL;  // device output, copied back for comparison
    cl_mem clInput   = NULL;    // device copy of input
    cl_mem clAverage = NULL;    // device output buffer

    // parse command line
    int platformIndex = 0;
    int deviceIndex   = 0;
    if (argc >= 2) platformIndex = atoi(argv[1]);
    if (argc >= 3) deviceIndex   = atoi(argv[2]);

    // get platforms (max 5)
    cl_int ciErr;
    cl_uint platforms = 0;
    cl_platform_id cpPlatform[5];
    ciErr = clGetPlatformIDs(5, cpPlatform, &platforms);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clGetPlatformIDs.\n", ciErr);
        return 1;
    }
    if (platformIndex > (int)platforms - 1) {
        fprintf(stderr, "Error: platform %i does not exist!\n", platformIndex);
        return 1;
    }

    // get devices
    cl_uint devices = 0;
    cl_device_id cdDevices[10];
    ciErr = clGetDeviceIDs(cpPlatform[platformIndex], CL_DEVICE_TYPE_ALL, 10, cdDevices, &devices);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clGetDeviceIDs.\n", ciErr);
        return 1;
    }
    if (deviceIndex > (int)devices - 1) {
        fprintf(stderr, "Error: device %i does not exist!\n", deviceIndex);
        return 1;
    }
    printDeviceInfo(cdDevices[deviceIndex]);

    // create context
    cl_context cxContext = clCreateContext(0, 1, &(cdDevices[deviceIndex]), NULL, NULL, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateContext.\n", ciErr);
        return 1;
    }

    // create queue
    cl_command_queue cqCommandQueue = clCreateCommandQueue(cxContext, cdDevices[deviceIndex], 0, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateCommandQueue.\n", ciErr);
        return 1;
    }

    // load OpenCL program
    char *cSource = load_program_source("kernel.cl");
    if (!cSource) {
        fprintf(stderr, "Error: Failed to load OpenCL program.\n");
        return 1;
    }

    // create program
    cl_program cpProgram = clCreateProgramWithSource(cxContext, 1, (const char **)&cSource, NULL, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateProgramWithSource.\n", ciErr);
        return 1;
    }

    // build the program
    ciErr = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error in clBuildProgram (%i).\n", ciErr);
        char *build_log;
        size_t ret_val_size;
        clGetProgramBuildInfo(cpProgram, cdDevices[deviceIndex], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = (char*) malloc((ret_val_size+1)*sizeof(*build_log));
        clGetProgramBuildInfo(cpProgram, cdDevices[deviceIndex], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0';
        fprintf(stderr, "%s\n", build_log);
        free(build_log);
        return 1;
    }

    // allocate and set host memory
    input       = (float*)malloc(SIZE * sizeof(input[0]));
    average     = (float*)malloc(SIZE * sizeof(average[0]));
    averageTest = (float*)malloc(SIZE * sizeof(averageTest[0]));
    for (int i = 0; i < SIZE; i++)
        input[i] = (float)rand() / float(RAND_MAX) + 0.1f;

    // allocate device buffers
    clInput = clCreateBuffer(cxContext, CL_MEM_READ_ONLY,  sizeof(cl_float) * SIZE, NULL, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateBuffer (input).\n", ciErr);
        return 1;
    }
    clAverage = clCreateBuffer(cxContext, CL_MEM_READ_WRITE, sizeof(cl_float) * SIZE, NULL, &ciErr);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clCreateBuffer (average).\n", ciErr);
        return 1;
    }

    // copy input to device
    ciErr = clEnqueueWriteBuffer(cqCommandQueue, clInput, CL_TRUE, 0,
                                 sizeof(cl_float) * SIZE, input, 0, NULL, NULL);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clEnqueueWriteBuffer.\n", ciErr);
        return 1;
    }
    clFinish(cqCommandQueue);

    // solve on CPU
    printf("Solving on CPU...\n");
    timeval timer;
    startTimer(&timer);
    solveCPU(average, input, SIZE);
    int time = stopTimer(&timer);
    printf("CPU performance: %f megavalues/s\n", (float)SIZE / (float)time);

    // solve on device
    printf("Solving on device...\n");
    startTimer(&timer);
    if (!solveDevice(cxContext, cqCommandQueue, cpProgram, clInput, clAverage, SIZE)) {
        fprintf(stderr, "Error in solveDevice.\n");
        return 1;
    }
    clFinish(cqCommandQueue);
    time = stopTimer(&timer);
    printf("Device performance: %f megavalues/s\n", (float)SIZE / (float)time);

    printf("Checking output...\n");

    // get device output
    ciErr = clEnqueueReadBuffer(cqCommandQueue, clAverage, CL_TRUE, 0,
                                sizeof(cl_float) * SIZE, averageTest, 0, NULL, NULL);
    if (ciErr != CL_SUCCESS) {
        fprintf(stderr, "Error %i in clEnqueueReadBuffer.\n", ciErr);
        return 1;
    }

    // release OpenCL objects
    clReleaseMemObject(clInput);
    clReleaseMemObject(clAverage);
    clReleaseProgram(cpProgram);
    clReleaseCommandQueue(cqCommandQueue);
    clReleaseContext(cxContext);

    // check results
    for (int i = 0; i < SIZE; i++) {
        if ((averageTest[i] != averageTest[i]) /* catches NaN */
            || (fabsf(average[i] - averageTest[i]) > 0.0001f)) {
            fprintf(stderr, "Data mismatch at index %i: %f should be %f :-(\n",
                    i, averageTest[i], average[i]);
            return 1;
        }
    }
    printf("Test OK.\n");

    free(input);
    free(average);
    free(averageTest);
    free(cSource);

    return 0;
}
