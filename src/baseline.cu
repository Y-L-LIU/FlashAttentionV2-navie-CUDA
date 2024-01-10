#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
// 定义 CUDA kernel 函数实现 self-attention
__global__ void selfAttention(float* K, float* Q, float* V, float* output, int r, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < r) {
        for (int i = 0; i < r; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < c; ++j) {
                sum += Q[idx * c + j] * K[i * c + j];
            }
            output[idx * c + i] = sum / sqrtf(static_cast<float>(c));
        }

        __syncthreads();

        for (int i = 0; i < r; ++i) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < c; ++j) {
                weighted_sum += output[idx * c + i] * V[i * c + j];
            }
            output[idx * c + i] = weighted_sum;
        }
    }
}

int main() {
    int r = 256; // 设置 r
    int c = 16; // 设置 c

    // 分配内存并初始化 K, Q, V 和输出张量
    float* h_K = new float[r * c];
    float* h_Q = new float[r * c];
    float* h_V = new float[r * c];
    float* h_output = new float[r * c];

    // 初始化 K, Q, V
    for (int i = 0; i < r * c; ++i) {
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
    // 在 GPU 上分配内存
    for (int i = 0; i < 100; i++) {
        float *d_K, *d_Q, *d_V, *d_output;
        cudaMalloc((void**)&d_K, sizeof(float) * r * c);
        cudaMalloc((void**)&d_Q, sizeof(float) * r * c);
        cudaMalloc((void**)&d_V, sizeof(float) * r * c);
        cudaMalloc((void**)&d_output, sizeof(float) * r * c);

        // 将数据从主机内存复制到 GPU 内存
        cudaMemcpy(d_K, h_K, sizeof(float) * r * c, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q, h_Q, sizeof(float) * r * c, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, sizeof(float) * r * c, cudaMemcpyHostToDevice);

        // 计算 grid 和 block 的大小
        int blockSize = 16;
        int gridSize = (r + blockSize - 1) / blockSize;

    // 调用 CUDA kernel 函数
    
        selfAttention<<<gridSize, blockSize>>>(d_K, d_Q, d_V, d_output, r, c);

        // 将结果从 GPU 复制回主机内存
        cudaMemcpy(h_output, d_output, sizeof(float) * r * c, cudaMemcpyDeviceToHost);

        // 释放 GPU 内存
        cudaFree(d_K);
        cudaFree(d_Q);
        cudaFree(d_V);
        cudaFree(d_output);
    }
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time for kernel execution: %.3f ms \n", milliseconds / 100);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
  
}
