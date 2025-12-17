#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (threadId < N && input[threadId] == K) {
        // IMPORTANT: Multiple threads may try to increment the same counter.
        // Using atomicAdd prevents race conditions by ensuring the increment
        // happens safely and correctly across all threads.
        atomicAdd(output, 1);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}