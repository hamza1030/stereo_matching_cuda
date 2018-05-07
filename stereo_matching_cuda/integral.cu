#include "integral.cuh"

void integral(float* image, float* integral, int width, int height) {
	float* d_image;
	float* d_integral;
	float* d_temp;
	const int w = width;
	const int h = height;
	memset(integral, 0, w*h * sizeof(float));
	CHECK(cudaMalloc((void**)&d_image, w*h*sizeof(float)));
	CHECK(cudaMalloc((void**)&d_integral, w*h * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_temp, w*h * sizeof(float)));

	CHECK(cudaMemcpy(d_image, image, w*h*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_integral, integral, w*h * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_temp, integral, w*h * sizeof(float), cudaMemcpyHostToDevice));


	dim3 threadsperblock(1024);
	dim3 blocknumbersRow((h+threadsperblock.x-1)/threadsperblock.x);
	dim3 blocknumbersCol((w + threadsperblock.x - 1) / threadsperblock.x);
	rowSum << <blocknumbersRow, threadsperblock >> > (d_image,d_temp,w,h);
	colSum << <blocknumbersCol, threadsperblock >> > (d_temp, d_integral, w, h);

	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());


	CHECK(cudaMemcpy(integral, d_integral, w*h * sizeof(float), cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_integral));
	CHECK(cudaFree(d_temp));
	CHECK(cudaFree(d_image));

	


}

/**
__global__ void transpose(float* in, float* out, const int w, const int h) {
	{
		__shared__ float temp[BLOCK_DIM][BLOCK_DIM + 1];
		int idx = blockIdx.x*BLOCK_DIM + threadIdx.x;
		int idy = blockIdx.y*BLOCK_DIM + threadIdx.y;
		if ((idx < w) && (idy < h))
		{
			int id_in = idy * w + idx;
			temp[threadIdx.y][threadIdx.x] = in[id_in];
		}
		__syncthreads();
		idx = blockIdx.y * BLOCK_DIM + threadIdx.x;
		idy = blockIdx.x * BLOCK_DIM + threadIdx.y;
		if ((idx < h) && (idy < w))
		{
			int id_out = idy * h + idx;
			out[id_out] = temp[threadIdx.x][threadIdx.y];
		}
	}
}
**/
__global__ void rowSum(float * image, float * integral, const int w, const int h)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	if (idy >= h) return;
	integral[idy*w] = image[idy*w];
	for (int idx = 1; idx < w; idx++)
	{
		
		integral[idy*w + idx] = image[idy*w + idx] + integral[idy*w + idx -1];
		__syncthreads();
	}
}

__global__ void colSum(float * image, float * integral, const int w, const int h) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= w) return;
	integral[idx] = image[idx];
	for (int idy = 1; idy < h; idy++)
	{
		integral[idy*w +idx] = image[idy*w + idx] + integral[(idy-1)*w + idx];
		__syncthreads();
	}

}
