#include "occlusion.cuh"

__global__ void detect_occlusionOnGPU(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h)
{
	int tdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tdx >= w * h) return;

	int d = (int)disparityLeft[tdx];
	int dprime = (int)disparityRight[tdx];
	if ((tdx % w) + d < 0 || (tdx % w) + d >= w || abs(d + dprime) > dLR)
		disparityLeft[tdx] = dOcclusion;
}

void detect_occlusion(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h)
{
	float* d_disparityLeft;
	float* d_disparityRight;

	int n = w * h;

	memset(disparityLeft, 0, n * sizeof(float));
	CHECK(cudaMalloc((void**)&d_disparityLeft, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_disparityRight, n * sizeof(float)));

	CHECK(cudaMemcpy(d_disparityLeft, disparityLeft, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_disparityRight, disparityRight, n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 nThreadsPerBlock(1024);
	dim3 nBlocks((n + nThreadsPerBlock.x - 1) / nThreadsPerBlock.x);

	detect_occlusionOnGPU << <nBlocks, nThreadsPerBlock >> > (d_disparityLeft, d_disparityRight, dOcclusion, dLR, w, h);

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(disparityLeft, d_disparityLeft, n * sizeof(float), cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_disparityLeft));
	CHECK(cudaFree(d_disparityRight));

}

/// Detect left-right discrepancies in disparity and put incoherent pixels to
/// value \a dOcclusion in \a disparityLeft.
void detect_occlusionOnCPU(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h)
{
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			int d = (int)disparityLeft[x + w * y];
			int dprime = (int)disparityRight[x + w * y];
			if (x + d < 0 || x + d >= w || abs(d + dprime) > dLR)
				disparityLeft[x + w * y] = dOcclusion;
		}
	}
}