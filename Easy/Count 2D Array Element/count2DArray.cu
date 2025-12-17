#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int threadId_x = blockDim.x * blockIdx.x + threadIdx.x;
    int threadId_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (threadId_x < M && threadId_y < N  && input[threadId_y * M + threadId_x] == K) {
        // IMPORTANT: Multiple threads may try to increment the same counter.
        // Using atomicAdd prevents race conditions by ensuring the increment
        // happens safely and correctly across all threads.
        atomicAdd(output, 1);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}