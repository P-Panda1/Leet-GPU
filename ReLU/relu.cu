#include <cuda_runtime.h>

__device__ float relu(float x) {
    return x < 0 ? 0 : x;
}


__global__ void relu_kernel(const float* input, float* output, int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < N) {
        output[threadId] = relu(input[threadId]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
