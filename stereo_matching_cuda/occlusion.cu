#include "occlusion.cuh"

__global__ void detect_occlusionOnGPU(float* disparityLeft, float* disparityRight, const int dOcclusion, const int w, const int h)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = id % w;
	if (id < w*h) {
	    int d = (int) disparityLeft[id];
		float diff = 1.0f*(D_LR + 1);
		if (idx + d >= 0 && idx + d < w) {
			diff = abs(d - disparityRight[id + d]);
		}
		disparityLeft[id] = (diff > D_LR) ? dOcclusion:d;

	}
	
}

void detect_occlusion(float* disparityLeft, float* disparityRight, const int dOcclusion, unsigned char* dmapl, unsigned char* dmapr, const int w, const int h)
{
	float* d_disparityLeft;
	float* d_disparityRight;
	unsigned char* d_dmapl;
	unsigned char* d_dmapr;


	int n = w * h;

	//detect_occlusionOnCPU(disparityLeft, disparityRight, dOcclusion, w, h);

	
	CHECK(cudaMalloc((void**)&d_disparityLeft, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_disparityRight, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_dmapl, n ));
	CHECK(cudaMalloc((void**)&d_dmapr, n));

	CHECK(cudaMemcpy(d_disparityLeft, disparityLeft, n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_disparityRight, disparityRight, n * sizeof(float), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(d_dmapl, dmapl, n , cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_dmapr, dmapr, n , cudaMemcpyHostToDevice));

	dim3 nThreadsPerBlock(1024);
	dim3 nBlocks((n + nThreadsPerBlock.x - 1) / nThreadsPerBlock.x);


	int minl = D_MIN;
	int maxl = D_MAX;
	int maxr = -1*D_MIN;
	int minr = -1*D_MAX;
	flToCh2OnGPU << <nBlocks, nThreadsPerBlock >> > (d_disparityLeft, d_dmapl, minl, maxl, n, dOcclusion);
	flToCh2OnGPU << <nBlocks, nThreadsPerBlock >> > (d_disparityRight, d_dmapr, minr, maxr, n, dOcclusion);
	detect_occlusionOnGPU << <nBlocks, nThreadsPerBlock >> > (d_disparityLeft, d_disparityRight, dOcclusion, w, h);
	

	CHECK(cudaMemcpy(disparityLeft, d_disparityLeft, n * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(dmapl, d_dmapl, n, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(dmapr, d_dmapr, n, cudaMemcpyDeviceToHost));

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	CHECK(cudaFree(d_disparityLeft));
	CHECK(cudaFree(d_disparityRight));
	CHECK(cudaFree(d_dmapl));
	CHECK(cudaFree(d_dmapr));

	//for (int i = 0; i < n; i++) {
	//	cout << i << " ResultGPU = " << disparityLeft[i] << endl;
	//}

	//float* h_disparityLeft = (float*)malloc(n * sizeof(float));
	//float* h_disparityRight = (float*)malloc(n * sizeof(float));
	//memcpy(h_disparityLeft, disparityLeft, n * sizeof(float));
	//memcpy(h_disparityRight, disparityRight, n * sizeof(float));

	//detect_occlusionOnCPU(h_disparityLeft, h_disparityRight, dOcclusion, w, h);
	//bool verif = check_errors(h_disparityLeft, disparityLeft, n);
	//if (verif) cout << "Disparity left ok!" << endl;
	//verif = check_errors(h_disparityRight, disparityRight, n);
	//if (verif) cout << "Disparity right ok!" << endl;

	//free(h_disparityLeft);
	//free(h_disparityRight);
}

/// Detect left-right discrepancies in disparity and put incoherent pixels to
/// value \a dOcclusion in \a disparityLeft.

void detect_occlusionOnCPU(float* disparityLeft, float* disparityRight, const int dOcclusion, const int w, const int h)
{
	int occlusion = 0;
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			int d = (int)disparityLeft[x + w * y];
			int dprime = (int)disparityRight[x + w * y + d];
			if (x + d < 0 || x + d >= w || abs(d + dprime) > D_LR)
			{
				occlusion++;
				disparityLeft[x + w * y] = dOcclusion;
			}
		}
	}
	cout << "occlusions: " << occlusion << endl;
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
__global__ void flToCh2OnGPU(float* image, unsigned char* result, int min, int max, int len, const int dOcclusion) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= len) { return;}
	float pix = image[i];
	float c = 1.0f*160*(pix - min)/(1.0f*(max -min));
	unsigned char val = (c > 255) ? 255 : (unsigned char)c;
	result[i] = val;
}