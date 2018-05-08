#include "rgb_to_grayscale.cuh"


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
	int i = channels * index;
	if (index < N)
	{
		double val = R_W * image[i] + G_W * image[i + 1] + B_W * image[i + 2];
		gray[index] = (unsigned char)val;
	}
}

unsigned char* rgb_to_grayscale(unsigned char* h_rgb, const int n, int channels, bool host_gpu_compare)
{
	const int nRGB = n * channels;

	unsigned char* h_gray;
	unsigned char* h_grayCPU;
	h_gray = (unsigned char *)malloc(n);
	h_grayCPU = (unsigned char *)malloc(n);
	memset(h_grayCPU, 0, n);
	memset(h_gray, 0, n);

	

	unsigned char* d_gray;
	unsigned char* d_rgb;

	// malloc device global memory
	CHECK(cudaMalloc((unsigned char**)&d_rgb, nRGB));
	CHECK(cudaMalloc((unsigned char**)&d_gray, n));

	CHECK(cudaMemcpy(d_rgb, h_rgb, nRGB, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_gray, h_gray, n, cudaMemcpyHostToDevice));

	int ilen = 64;
	dim3 blockDim(ilen);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
	sumArraysOnGPU << <gridDim, blockDim >> > (d_rgb, d_gray, n, channels);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(h_gray, d_gray, n, cudaMemcpyDeviceToHost));
	
	if (host_gpu_compare)
	{
		sumArraysOnHost(h_rgb, h_grayCPU, n, channels);
		bool verif = check_errors_grayscale(h_grayCPU, h_gray,n);
		if (verif) cout << "RGB to grayscale ok!" << endl;
	}
	// free device global memory
	CHECK(cudaFree(d_gray));
	CHECK(cudaFree(d_rgb));
	free(h_grayCPU);


	return h_gray;
}

__host__ bool check_errors_grayscale(unsigned char * host, unsigned char * gpu, int len) {
	bool res = true;
	for (int i = 0; i < len; i++) {
		if (gpu[i] != host[i]) {
			res = false;
		}
	}
	return res;
}