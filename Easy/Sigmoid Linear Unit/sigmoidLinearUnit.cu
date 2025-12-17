#include <cuda_runtime.h>

__device__ float silu(float value) {
    float res = value/(1+__expf(-value)); 
    return res;
}

__global__ void silu_kernel(const float* input, float* output, int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < N) {
        output[threadId] = silu(input[threadId]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}