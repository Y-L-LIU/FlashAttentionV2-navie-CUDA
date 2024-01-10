#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

__global__ void flashAttention(float* Q, float* K, float* V, float* O, int d, int N)
{
    int x = threadIdx.x;

	int i = blockIdx.x;
	// Max shared memory per block: 49152 bytes(48KB)
	// Number of floats that can be stored in shared memory: 12288	, 12288/(64*4)=48, 32 is ok
    // batch_size=4 when dim=512, batch_size=8 when dim=256. batch_size=16 when dim=256. batch_size=32 when dim=64
    int batch_size = 4;
    int batch_num = N//B_r;
	__shared__ float Kj[batch_size * d];
	__shared__ float Vj[batch_size * d];
	__shared__ float Qi[batch_size * d];
	__shared__ float Oi[batch_size * d];
	__shared__ float temp[batch_size * d];

	__shared__ float Sij[batch_size * batch_size];

	__shared__ float mi[batch_num * batch_size];
	__shared__ float li[batch_num * batch_size];

	__shared__ float lij[batch_size];
	__shared__ float mij[batch_size];

	__shared__ float li_new[batch_num * batch_size];
	__shared__ float mi_new[batch_num * batch_size];

    float l_new = 0;
	
	for (int ind = x; q < batch_num * batch_size; q += batch_size) {
		mi[q] = -INFINITY;
		mi_new[q] = -INFINITY;
		li[q] = 0;
		li_new[q] = 0;
	}
}