#include "integral.cuh"

void integral(float* image, float* integral, int width, int height) {
	float* d_image;
	float* d_integral;
	float* d_integralT;
	float* d_integralT2;
	float* d_integralT3;
	const int w = width;
	const int h = height;
	memset(integral, 0, w*h * sizeof(float));
	CHECK(cudaMalloc((void**)&d_image, w*h * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_integral, w*h * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_integralT, w*h * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_integralT2, w*h * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_integralT3, w*h * sizeof(float)));

	CHECK(cudaMemcpy(d_image, image, w*h * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_integral, integral, w*h * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_integralT, integral, w*h * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_integralT2, integral, w*h * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_integralT3, integral, w*h * sizeof(float), cudaMemcpyHostToDevice));


	dim3 threadsperblock(1024);
	dim3 blocknumbersRow((h + threadsperblock.x - 1) / threadsperblock.x);
	dim3 blocknumbersCol((w + threadsperblock.x - 1) / threadsperblock.x);
	dim3 threadsT(B_SIZE, B_SIZE, 1);
	dim3 blocksT((w + B_SIZE -1) / B_SIZE, (h + B_SIZE -1)/ B_SIZE , 1);
	dim3 blocksT2((h + B_SIZE - 1) / B_SIZE, (w + B_SIZE - 1) / B_SIZE, 1);
	rowSum << <blocknumbersRow, threadsperblock >> > (d_image, d_integralT, w, h);
	transpose << <blocksT, threadsT >> > (d_integralT, d_integralT2, w, h);
	rowSum << <blocknumbersCol, threadsperblock >> > (d_integralT2, d_integralT3, h, w);
	dim3 threadsT2(B_SIZE, B_SIZE, 1);
	transpose << <blocksT2, threadsT >> > (d_integralT3, d_integral, h, w);
	//rowSum << <blocknumbersRow, threadsperblock >> > (d_image, d_integralT, w, h);
	//colSum << <blocknumbersCol, threadsperblock >> > (d_integralT, d_integral, w, h);

	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(integral, d_integral, w*h * sizeof(float), cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_integral));
	CHECK(cudaFree(d_integralT));
	CHECK(cudaFree(d_integralT2));
	CHECK(cudaFree(d_integralT3));
	CHECK(cudaFree(d_image));
}

__global__ void transpose(float* in, float* out, const int w, const int h) {
	{
		__shared__ float temp[B_SIZE][B_SIZE + 1];
		int idx = blockIdx.x*B_SIZE + threadIdx.x;
		int idy = blockIdx.y*B_SIZE + threadIdx.y;
		if ((idx < w) && (idy < h))
		{
			int id_in = idy * w + idx;
			temp[threadIdx.y][threadIdx.x] = in[id_in];
		}
		__syncthreads();
		idx = blockIdx.y * B_SIZE + threadIdx.x;
		idy = blockIdx.x * B_SIZE + threadIdx.y;
		if ((idx < h) && (idy < w))
		{
			int id_out = idy * h + idx;
			out[id_out] = temp[threadIdx.x][threadIdx.y];
		}
	}
} 

	



__global__ void rowSum(float * in, float * out, const int w, const int h)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	if (idy >= h) return;
	out[idy*w] = in[idy*w];
	for (int idx = 1; idx < w; idx++)
	{
		out[idy*w + idx] = in[idy*w + idx] + out[idy*w + idx - 1];

		__syncthreads();

	}
}

void integralOnCPU(float * in, float * out, const int w, const int h)
{
	float* temp = new float[w * h];
	memset(temp, 0.0f, w * h);

	for (int y = 0; y < h; y++)
	{
		temp[y*w] = in[y*w];

		for (int x = 1; x < w; x++)
		{
			temp[y*w + x] = in[y*w + x] + temp[y*w + x - 1];
		}
	}


	for (int x = 0; x < w; x++)
	{
		out[x] = temp[x];

		for (int y = 1; y < h; y++)
		{
			out[y*w + x] = temp[y*w + x] + out[(y - 1)*w + x];
		}
	}

	free(temp);
}

__global__ void colSum(float * in, float * out, const int w, const int h) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= w) return;
	out[idx] = in[idx];
	for (int idy = 1; idy < h; idy++)
	{
		out[idy*w + idx] = in[idy*w + idx] + out[(idy - 1)*w + idx];
		__syncthreads();

	}
}

