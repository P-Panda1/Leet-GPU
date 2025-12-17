#include <cuda_runtime.h>

__device__ float swiglu(float value1, float value2) {
    return value2 * (value1/(1+__expf(-value1)));
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < halfN) {
        float x1 = input[threadId]; 
        float x2 = input[threadId+halfN];
        output[threadId] = swiglu(x1, x2);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}