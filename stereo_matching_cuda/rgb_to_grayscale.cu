#include "rgb_to_grayscale.cuh"
#include "Clock.h"

void sumArraysOnHost(unsigned char* image, unsigned char* gray, const int N, int channels)
{
	for (int idx = 0; idx < N; idx++)
	{
		int i = channels * idx;
		double val = R_W * image[i] + G_W * image[i + 1] + B_W * image[i + 2];
		gray[idx] = (unsigned char)val;
	}
}

__global__ void sumArraysOnGPU(unsigned char* image, unsigned char* gray, const int N, int channels)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//N = (4180*2160)
	int i = channels * index;
	if (index < N)
	{
		double val = R_W * image[i] + G_W * image[i + 1] + B_W * image[i + 2];
		gray[index] = (unsigned char)val;
	}
}

unsigned char* rgb_to_grayscale(unsigned char* h_rgb, const int n, int channels, bool cuda)
{
	const int nRGB = n * channels;

	unsigned char* h_gray;
	unsigned char* h_grayCPU;
	h_gray = (unsigned char *)malloc(n);
	h_grayCPU = (unsigned char *)malloc(n);

	memset(h_grayCPU, 0, n);
	memset(h_gray, 0, n);

	if (!cuda)
	{
		Clock clock;
		clock.init();
		sumArraysOnHost(h_rgb, h_grayCPU, n, channels);
		clock.getTotalTime();
		return h_grayCPU;
	}

	unsigned char* d_gray;
	unsigned char* d_rgb;

	// malloc device global memory
	CHECK(cudaMalloc((unsigned char**)&d_rgb, nRGB));
	CHECK(cudaMalloc((unsigned char**)&d_gray, n));

	CHECK(cudaMemcpy(d_rgb, h_rgb, nRGB, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_gray, h_gray, n, cudaMemcpyHostToDevice));

	int ilen = 1024;
	dim3 blockDim(ilen);
	cout << " " << blockDim.x << endl;
	int blockdimx = 1024;
	if (blockDim.x < blockdimx) blockdimx = blockDim.x;
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
	Clock clock;
	clock.init();
	sumArraysOnGPU << <gridDim, blockDim >> > (d_rgb, d_gray, n, channels);
	clock.getTotalTime();
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(h_gray, d_gray, n, cudaMemcpyDeviceToHost));

	// free device global memory
	CHECK(cudaFree(d_gray));
	CHECK(cudaFree(d_rgb));

	return h_gray;
}