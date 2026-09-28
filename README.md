# OpenCL Parallel Sliding-Window Average

GPU-accelerated implementation of a sliding-window average over a large
floating-point vector using OpenCL.

The project was developed as an OpenCL parallel computing project and evaluated
on the University of Vienna ALMA computing cluster using NVIDIA Tesla K20m GPUs.

## Problem

For every element `i` of the input vector, the program computes the average of
a fixed-size window centered around that element:

```text
average[i] = (1 / (2R)) * Σ input[i + j]
                         j=-R ... R-1
```

where `R` is half of the window size.

When the window extends beyond the input boundaries, the nearest boundary
element is used.

The project uses:

- `R = 128`
- window size = `256`
- input size = `33,554,432` floating-point values

## Implementation

The project contains both a sequential CPU reference implementation and a
parallel OpenCL implementation.

### CPU Reference

`kernel_C.cpp` implements the sequential version of the computation. Its
results are used to validate the OpenCL implementation.

### OpenCL Kernel

`kernel.cl` implements the GPU computation, while `kernel_exec.cpp` handles
kernel creation, argument setup, and execution.

The implementation uses:

- 256 work-items per work-group
- local memory for data shared between work-items
- halo loading for neighboring window elements
- work-group synchronization with `barrier`
- `float4` vectorized accumulation
- `vload4`
- loop unrolling

Each work-group loads the input values required by its work-items into local
memory, including the additional halo elements required by the sliding window:

```c
__local float sh[WG_SIZE + WINDOW];
```

With `WG_SIZE = 256` and `WINDOW = 256`, each work-group uses 512 local
floating-point values.

## Correctness Validation

The host framework computes the result using both the CPU reference
implementation and the OpenCL implementation.

The GPU result is copied back to the host and compared against the CPU result
using an absolute error tolerance of `0.0001`.

The framework reports `Test OK.` when all output values satisfy the required
tolerance.

## Performance

The framework measures CPU and OpenCL execution throughput in megavalues per
second.

The optimized implementation was evaluated with:

- `R = 128`
- `SIZE = 33,554,432`
- NVIDIA Tesla K20m

The project specification required a performance of at least
`1,000 megavalues/s` on the K20m for the maximum performance score.

## Compilation

The original project was compiled on the ALMA cluster using:

```bash
g++ -O3 -o framework framework.cpp \
-I/usr/local/cuda/targets/x86_64-linux/include/ \
-L/usr/local/cuda/targets/x86_64-linux/lib/ \
-lOpenCL
```

## Execution

The executable accepts an OpenCL platform ID and device ID:

```bash
./framework <platform> <device>
```

For example:

```bash
./framework 0 0
```

executes the program on platform `0`, device `0`.

The available platforms and devices depend on the OpenCL environment of the
machine on which the program is executed.

## Project Structure

```text
OpenCL-Parallel-Computing/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── framework.cpp
├── kernel_C.cpp
├── kernel_exec.cpp
└── kernel.cl
```

| File | Description |
|---|---|
| `framework.cpp` | Host framework, OpenCL initialization, data management, validation, and benchmarking |
| `kernel_C.cpp` | Sequential CPU reference implementation |
| `kernel_exec.cpp` | OpenCL kernel creation, argument setup, and execution |
| `kernel.cl` | Parallel OpenCL kernel |
| `CMakeLists.txt` | CMake project configuration |

## Technologies

- C++
- OpenCL
- OpenCL C
- GPU parallel computing
- Local memory and work-group synchronization
- SIMD/vectorized computation

## Computing Environment

The project was designed to run on the University of Vienna ALMA computing
cluster.

The cluster provides NVIDIA Tesla K20m GPUs for OpenCL computation. Access to
the cluster is provided through:

```text
alma.par.univie.ac.at
```

from the University network or through the University of Vienna VPN.

## Notes

The project framework was provided as part of the OpenCL project assignment.
The main implementation work consisted of developing the OpenCL kernel in
`kernel.cl` and the kernel execution code in `kernel_exec.cpp`.

The framework is responsible for input generation, device initialization,
host/device data transfers, CPU reference computation, correctness checking,
and performance measurement.