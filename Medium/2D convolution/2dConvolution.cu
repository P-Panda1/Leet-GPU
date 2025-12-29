#include <cuda_runtime.h>


__global__ void conv(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {

    int threadId_x = blockDim.x * blockIdx.x + threadIdx.x;
    int threadId_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int out_cols = input_cols - kernel_cols + 1;
    int out_rows = input_rows - kernel_rows + 1;
    
    if (threadId_x < out_cols && threadId_y <out_rows){
        float sum = 0;

        for (int i=0; i<kernel_cols; i++) {
            for (int j=0; j<kernel_rows; j++) {
                sum += input[(threadId_y + j) * input_cols + (threadId_x + i)] * kernel[j * kernel_cols + i];
            }
        }
        output[threadId_y * out_cols + threadId_x] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) 
{
    dim3 threads(16, 16);

    dim3 blocks(
        ((input_cols - kernel_cols + 1) + threads.x - 1) / threads.x,
        ((input_rows - kernel_rows + 1) + threads.y - 1) / threads.y
        );
    
    conv<<<blocks, threads>>>(input, kernel, output, input_rows,
                    input_cols, kernel_rows, kernel_cols);   

}