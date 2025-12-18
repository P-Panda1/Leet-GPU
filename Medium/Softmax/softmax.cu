#include <cuda_runtime.h>

__global__ void find_max(const float* input, float* dmax, int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < N) {
        atomicMax((int*)dmax, __float_as_int(input[threadId]));
    }
}

__global__ void exponent_and_sum(const float* input, float* output, float* dsum, float* dmax, int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < N) {
        // Create a float val to handle both cases
        float val = __expf(input[threadId] - *dmax);

        // Adding the non normalised values 
        output[threadId] = val;
        atomicAdd(dsum, val);
    }
}

__global__ void softmax_kernel(float* output, float* dsum, int N) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < N) {
        // dsum is a pointer to ONE float in global memory
        output[threadId] /= *dsum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Setting up the device memory for sum
    float* dsum;
    cudaMalloc(&dsum, sizeof(float));

    // Setting up the device memory for max
    float* dmax;
    cudaMalloc(&dmax, sizeof(float));

    // IMPORTANT: initialize sum to zero
    cudaMemset(dsum, 0, sizeof(float));

    // Setting one value from input
    // We choose one value from input so that the value is within the input domain
    // We can't just initialise from 0 as it will become max if all values are 0
    cudaMemcpy(dmax, input, sizeof(float), cudaMemcpyDeviceToDevice);

    // Find the max in the input domain

    // Exponentiating and summing all the values
    exponent_and_sum<<<blocksPerGrid, threadsPerBlock>>>(input, output, dsum, dmax, N);

    // Softmaxing using the updated output thread
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, dsum, N);
    cudaDeviceSynchronize();
    cudaFree(dsum);
}