#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    char max_color = 255;
    if (threadId < (width * height)){
        image[threadId*4] = max_color - image[threadId*4];
        image[threadId*4+1] = max_color - image[threadId*4+1];
        image[threadId*4+2] = max_color - image[threadId*4+2];  
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}