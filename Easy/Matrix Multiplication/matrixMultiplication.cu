#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int threadId_i = blockIdx.x * blockDim.x + threadIdx.x; //Column index
    int threadId_j = blockIdx.y * blockDim.y + threadIdx.y; //Row index
    int currId = threadId_j * K + threadId_i; // Essentially current ID
    
    if ((threadId_i < K) && (threadId_j < M)) {
        float currId_value = 0;
        for (int n=0; n<N; n++) {
            // Iterate over the number of rows that A matrix has or number of columns that B matrix has
            currId_value += A[threadId_j*N + n] * B[n*K + threadId_i];
            // threadId_i gives the coulmn value of A that we currently look at
            // threadId_j gives the column value of B that we currently look at
        }
        C[currId] = currId_value;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
