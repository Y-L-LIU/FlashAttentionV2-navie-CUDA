#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// attention_kernel.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <algorithm>

using namespace std;

#define BLOCK_DIM 16
#define N 256
#define d 16
#define num_block_x (N+ BLOCK_DIM -1)/ BLOCK_DIM
#define num_block_y (d+ BLOCK_DIM -1)/ BLOCK_DIM

__global__ void flashAttention(float* Q, float* K, float* V, float* O)
{
	#include <math.h>
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// Max shared memory per block: 49152 bytes(48KB)
	// Number of floats that can be stored in shared memory: 12288	, 12288/(64*4)=48, 32 is ok
	// batch_size=4 when dim=512, batch_size=8 when dim=256. batch_size=16 when dim=256. batch_size=32 when dim=64
	// const int batch_size = 4;

	// int batch_num = ceil(N/ num_block_x);
	// __shared__ float block_sum[num_block_x][num_block_y];
	__shared__ float block_max[BLOCK_DIM][BLOCK_DIM];
	__shared__ float S[BLOCK_DIM][BLOCK_DIM];
	// __shared__ float global_sum[num_block_x][num_block_y];
	__shared__ float global_max[BLOCK_DIM][BLOCK_DIM];
	__shared__ float global_max_old[BLOCK_DIM][BLOCK_DIM];
	float L;
	float L_old;
	float multi_o;
	__shared__ float O_temp[BLOCK_DIM][BLOCK_DIM];
	// global_sum[threadIdx.x][threadIdx.y] = 0.0f;
	global_max[threadIdx.x][threadIdx.y] = -INFINITY;
	global_max_old[threadIdx.x][threadIdx.y] = -INFINITY;
	// block_sum[threadIdx.x][threadIdx.y] = 0.0f;
	block_max[threadIdx.x][threadIdx.y] = -INFINITY;

	L = L_old = 0.0f;
	// Here we did not use the GEMM
	for (int batch_n = 0; batch_n < num_block_x; batch_n++) {
		if (x < N && y < N) {
			float S_ij = 0.0f;
			for (int idx = 0; idx < d; idx++) {
				S_ij += Q[x * d + idx] * K[y * d + idx];
			}
			S[threadIdx.x][threadIdx.y] = S_ij;
			// block_sum[threadIdx.x][threadIdx.y] = 1.0f;
			block_max[threadIdx.x][threadIdx.y] = S_ij;
		}
		else {
			S[threadIdx.x][threadIdx.y] = 0.0f;
			// block_sum[threadIdx.x][threadIdx.y] = 0.0f;
			block_max[threadIdx.x][threadIdx.y] = -INFINITY;
		}
		__syncthreads();
		// case 1:
		//		m_i0 = max(S_ij)

		// case 2:
		//		m_ij = max{m_i,j-1, max(S_ij)}
		for (int step = num_block_x / 2; step > 0; step /= 2) {
			if (threadIdx.x < step) {
				if (block_max[threadIdx.x][threadIdx.y] < block_max[threadIdx.x + step][threadIdx.y]) {
					block_max[threadIdx.x][threadIdx.y] = block_max[threadIdx.x + step][threadIdx.y];
				}
			}
			__syncthreads();
		}


		__syncthreads();
		if (batch_n * BLOCK_DIM + threadIdx.x < N && y < d) {
			if (global_max[threadIdx.x][threadIdx.y] < block_max[0][threadIdx.y]) {
				global_max[threadIdx.x][threadIdx.y] = block_max[0][threadIdx.y];
			}
		}
		__syncthreads();
		// P_ij = exp(S_ij - m_ij)
		if (batch_n * BLOCK_DIM + threadIdx.x < N && y < d) {
			S[threadIdx.x][threadIdx.y] = exp(S[threadIdx.x][threadIdx.y] - global_max[threadIdx.x][threadIdx.y]);
		}
		else {
			S[threadIdx.x][threadIdx.y] = 0.0f;
		}
		__syncthreads();

		// O_ij = diag(exp(m_i,j-1-m_i,j))O_i,j-1 + P_ij * V_j
		// L_ij = exp(m_i,j-1-m_i,j)*L_i,j-1 + \rowsum(P_ij)
		if (y < d) {
			multi_o = 0.0f;
			for (int idx = 0; idx < BLOCK_DIM; idx++) {
				if ( (idx + batch_n * BLOCK_DIM)  < N) {
					multi_o += S[threadIdx.x][idx] * V[(idx + batch_n* BLOCK_DIM) * d + y];
				}
			}
			//multi_o = 1.0;
			if (batch_n == 0) {
				O_temp[threadIdx.x][threadIdx.y] = multi_o;
				for (int idx = 0; idx < num_block_y; ++idx) {
					L += S[threadIdx.x][idx];
				}

			}
			else {
				O_temp[threadIdx.x][threadIdx.y] = multi_o +
					exp(global_max_old[threadIdx.x][threadIdx.y] - global_max[threadIdx.x][threadIdx.y]) * O_temp[threadIdx.x][threadIdx.y];
				for (int idx = 0; idx < num_block_y; ++idx) {
					L += S[threadIdx.x][idx];
				}
				L += L_old * exp(global_max_old[threadIdx.x][threadIdx.y] - global_max[threadIdx.x][threadIdx.y]);
			}
			global_max_old[threadIdx.x][threadIdx.y] = global_max[threadIdx.x][threadIdx.y];
			L_old = L;
		}
		else {
			global_max_old[threadIdx.x][threadIdx.y] = -INFINITY;
			L_old = 0.0f;
		}
	}
	O[x * d + y] = float(O_temp[threadIdx.x][threadIdx.y] / L);
}

void flashAttentionLauncher(float* Q, float* K, float* V, float* O) {

	float* d_Q;
	float* d_K;
	float* d_V;
	float* d_O;

	cudaMalloc((void**)&d_Q, N * d * sizeof(float));
	cudaMalloc((void**)&d_K, N * d * sizeof(float));
	cudaMalloc((void**)&d_V, N * d * sizeof(float));
	cudaMalloc((void**)&d_O, N * d * sizeof(float));

	cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_O, O, N * d * sizeof(float), cudaMemcpyHostToDevice);

	dim3 grid_dim(num_block_x, num_block_y, 1);
	dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
	// int share_mem = 2 * BLOCK_DIM * BLOCK_DIM * sizeof(float);

	flashAttention << <grid_dim, block_dim >> > (d_Q, d_K, d_V, d_O);
	cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);
	//print output

}



void initializeRandomMatrix(float* matrix, int rows, int cols) {
	// Seed the random number generator
	srand(static_cast<unsigned int>(time(nullptr)));

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// Assign random float values to the matrix
			matrix[i * cols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		}
	}
}

int main()
{
	//test scaled_dot_product_attention
	float* q = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));
	float* k = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));
	float* v = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));
	float* output = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));
	// printf('%f', num_block_x);
	if (q == NULL || k == NULL || v == NULL || output == NULL) {
		printf("Memory allocation failed");

		// Free any allocations that were successful
		free(q);
		free(k);
		free(v);
		free(output);

		return -1;
	}

	initializeRandomMatrix(q, N, d);
	initializeRandomMatrix(k, N, d);
	initializeRandomMatrix(v, N, d);

	// for (int i = 0; i < N * d; i++)
	// {
	// 	q[i] = 1.0f;
	// 	k[i] = 2.0f;
	// 	v[i] = 12.0f;
	// 	output[i] = 0.0f;
	// };

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		flashAttentionLauncher(q, k, v, output);
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