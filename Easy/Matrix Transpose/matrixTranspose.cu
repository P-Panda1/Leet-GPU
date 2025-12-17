#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int threadId_x = blockDim.x * blockIdx.x + threadIdx.x;
    int threadId_y = blockDim.y * blockIdx.y + threadIdx.y;

    // Ensure you give the condition for threadId_x against cols 
    // Ensure you give the condition for threadId_y against rows
    // Not doing so will have CUDA check floating point for values outside of desired values and will check for wrong size of the matrix
    if ((threadId_x < cols) && (threadId_y < rows)) {
        output[threadId_x * rows + threadId_y] = input[threadId_y * cols + threadId_x];
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}