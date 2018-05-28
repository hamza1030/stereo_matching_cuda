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

	//memset(disparityLeft, 0, n * sizeof(float));
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

// Filling

void fill_occlusion(float* disparity, const int w, const int h, const float vMin)
{
	float* d_disparity;

	int n = w * h;

	CHECK(cudaMalloc((void**)&d_disparity, n * sizeof(float)));

	CHECK(cudaMemcpy(d_disparity, disparity, n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 nThreadsPerBlock(1024);
	dim3 nBlocks((n + nThreadsPerBlock.x - 1) / nThreadsPerBlock.x);

	fill_occlusionOnGPU1 << <nBlocks, nThreadsPerBlock >> > (d_disparity, w, h, vMin);

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(disparity, d_disparity, n * sizeof(float), cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_disparity));
}

__global__ void fill_occlusionOnGPU1(float* disparity, const int w, const int h, const float vMin)
{
	int tdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tdx >= w * h) return;

	int dX = disparity[tdx];
	// ignore non-occluded pixels
	if (dX >= vMin) return;

	int y = tdx / w;

	int xLeft = tdx % w;
	float dLeft = vMin;

	// find first non occluded left pixel
	while (xLeft >= 0)
	{
		if (disparity[xLeft + w * y] >= vMin) {
			dLeft = disparity[xLeft + w * y];
			break;
		}

		xLeft -= 1;
	}

	int xRight = tdx % w;
	float dRight = vMin;

	// find first non occluded right pixel
	while (xRight < w)
	{
		// if it is nonoccluded, stop
		if (disparity[xRight + w * y] >= vMin) {
			dRight = disparity[xRight + w * y];
			break;
		}

		xRight += 1;
	}

	disparity[tdx] = max(dLeft, dRight);
}


__global__ void fill_occlusionOnGPU2(float* disparity, const int w, const int h, const float vMin)
{
	// 1) threads computing dLeft
	// 2) threads computing dRight
	// syncthreads
	// max(dLeft, dRight)
}

/// Fill pixels below value vMin using maximum value between the two closest (at left and at right) pixels on same
/// line above vMin.
void fill_occlusionOnCPU(float* disparity, const int w, const int h, const float vMin) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++)
		{
			int dX = disparity[x + w * y];
			// ignore non-occluded pixels
			if (dX >= vMin) continue;

			int xLeft = x;
			float dLeft = vMin;

			// find first non occluded left pixel
			while (xLeft >= 0)
			{
				if (disparity[xLeft + w * y] >= vMin) {
					dLeft = disparity[xLeft + w * y];
					break;
				}

				xLeft -= 1;
			}

			int xRight = x;
			float dRight = vMin;

			// find first non occluded right pixel
			while (xRight < w)
			{
				// if it is nonoccluded, stop
				if (disparity[xRight + w * y] >= vMin) {
					dRight = disparity[xRight + w * y];
					break;
				}

				xRight += 1;
			}

			disparity[x + w * y] = max(dLeft, dRight);
		}
	}
}
