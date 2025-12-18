#include <cuda_runtime.h>

__global__ void solver(const float* input, float* output, int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < N) {
        atomicAdd(output, input[threadId]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    solver<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}