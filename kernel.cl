#define R 128
#define WINDOW (2 * R)
#define WG_SIZE 256

__kernel void compute(__global const float* input,
                      __global float* average,
                      int size) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int global_start = group_id * WG_SIZE;
    int global_i = global_start + local_id;

    __local float sh[WG_SIZE + WINDOW];

    int base = global_start - R;

    int idx1 = base + local_id;
    if (idx1 < 0) idx1 = 0;
    if (idx1 >= size) idx1 = size - 1;
    sh[local_id] = input[idx1];

    if (local_id < WINDOW) {
        int idx2 = base + local_id + WG_SIZE;
        if (idx2 < 0) idx2 = 0;
        if (idx2 >= size) idx2 = size - 1;
        sh[local_id + WG_SIZE] = input[idx2];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 sum4 = (float4)(0.0f);

    #pragma unroll
    for (int w = 0; w < WINDOW; w += 4) {
        sum4 += vload4(0, &sh[local_id + w]);
    }
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;

    average[global_i] = sum / (float)WINDOW;
}