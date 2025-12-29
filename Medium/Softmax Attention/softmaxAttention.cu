#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void init_max(float* dmax, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) dmax[i] = -INFINITY;
}

__global__ void qk_kernel(
    const float* Q, const float* K, float* S, float* dmax,
    int M, int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.f;
        for (int i = 0; i < d; i++)
            sum += Q[row * d + i] * K[col * d + i];

        sum /= sqrtf((float)d);
        S[row * N + col] = sum;
        atomicMaxFloat(&dmax[row], sum);
    }
}

__global__ void exponent_and_sum(
    float* S, float* dsum, const float* dmax,
    int M, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float v = expf(S[row * N + col] - dmax[row]);
        S[row * N + col] = v;
        atomicAdd(&dsum[row], v);
    }
}

__global__ void pv_kernel(
    const float* S, const float* V, const float* dsum,
    float* output, int M, int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < d) {
        float acc = 0.f;
        for (int i = 0; i < N; i++)
            acc += S[row * N + i] * V[i * d + col];

        output[row * d + col] = acc / dsum[row];
    }
}

extern "C" void solve(
    const float* Q, const float* K, const float* V,
    float* output, int M, int N, int d
) {
    dim3 threads(16, 16);

    dim3 blocksQK(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    dim3 blocksPV(
        (d + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    float *S, *dsum, *dmax;
    cudaMalloc(&S, M * N * sizeof(float));
    cudaMalloc(&dsum, M * sizeof(float));
    cudaMalloc(&dmax, M * sizeof(float));

    cudaMemset(dsum, 0, M * sizeof(float));
    init_max<<<(M + 255) / 256, 256>>>(dmax, M);

    qk_kernel<<<blocksQK, threads>>>(Q, K, S, dmax, M, N, d);
    cudaDeviceSynchronize();

    exponent_and_sum<<<blocksQK, threads>>>(S, dsum, dmax, M, N);
    cudaDeviceSynchronize();

    pv_kernel<<<blocksPV, threads>>>(S, V, dsum, output, M, N, d);

    cudaFree(S);
    cudaFree(dsum);
    cudaFree(dmax);
}