#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < N) {
        float temp = input[threadId];
        input[threadId] = input[N - threadId -1];
        input[N - threadId - 1] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}