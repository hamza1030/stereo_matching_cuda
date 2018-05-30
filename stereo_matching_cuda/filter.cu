#include "filter.cuh"

/**
__host__ void boxFilterOnCpu(char*in, char* out, int radius)
{
	double* S = new double[w*h]; // Use double to mitigate precision loss
	for (int i = w * h - 1; i >= 0; i--)
		S[i] = static_cast<double>(tab[i]);

	//cumulative sum table S, eq. (24)
	for (int y = 0; y<h; y++) { //horizontal
		double *in = S + y * w, *out = in + 1;
		for (int x = 1; x<w; x++)
			*out++ += *in++;
	}
	for (int y = 1; y<h; y++) { //vertical
		double *in = S + (y - 1)*w, *out = in + w;
		for (int x = 0; x<w; x++)
			*out++ += *in++;
	}

	//box filtered image B
	Image B(w, h);
	float *out = B.tab;
	for (int y = 0; y<h; y++) {
		int ymin = std::max(-1, y - radius - 1);
		int ymax = std::min(h - 1, y + radius);
		for (int x = 0; x<w; x++, out++) {
			int xmin = std::max(-1, x - radius - 1);
			int xmax = std::min(w - 1, x + radius);
			// S(xmax,ymax)-S(xmin,ymax)-S(xmax,ymin)+S(xmin,ymin), eq. (25)
			double val = S[ymax*w + xmax];
			if (xmin >= 0)
				val -= S[ymax*w + xmin];
			if (ymin >= 0)
				val -= S[ymin*w + xmax];
			if (xmin >= 0 && ymin >= 0)
				val += S[ymin*w + xmin];
			*out = static_cast<float>(val / ((xmax - xmin)*(ymax - ymin))); //average
		}
	}
	delete[] S;
	return B;
}
**/
/**

__host__ void covarianceOnCpu(char* I, char* out, int radius, char* mean) {
	var = boxFilter(I*I, char* out, radius);

	return boxFilter(r) - mean1 * mean2;
}
**/

__global__ void boxFilterOnGPU(unsigned char* image, unsigned char* mean, int width, int height) {
	int i = blockIdx.x * TILE_WIDTH + threadIdx.x - RADIUS;
	int j = blockIdx.y * TILE_HEIGHT + threadIdx.y - RADIUS;
	int ind = j * width + i;
	__shared__ float sharedMem[B_W][B_H];
	if (i < 0 || j < 0 || i >= width || j >= height) {
		sharedMem[threadIdx.x][threadIdx.y] = 0;
		return;
	}
	sharedMem[threadIdx.x][threadIdx.y] = image[ind];
	

	__syncthreads();

	// box filter (only for threads inside the tile)
	if ((threadIdx.x >= RADIUS) && (threadIdx.x < (B_W - RADIUS)) && (threadIdx.y >= RADIUS) && (threadIdx.y < (B_H - RADIUS))) {
		float sum = 0;
		for (int ix = -RADIUS; ix <= RADIUS; ix++) {
			for (int iy = -RADIUS; iy <= RADIUS; iy++) {
				sum += sharedMem[threadIdx.x + ix][threadIdx.y + iy];
			}
		}
		int val = sum / ((2 * RADIUS + 1)*(2 * RADIUS + 1));
		mean[ind] = (unsigned char)val;
		//mean[ind] = 220;
	}
}
__global__ void boxFilterfloatOnGpu(float* image, float* mean, int width, int height) {
	int i = blockIdx.x * TILE_WIDTH + threadIdx.x - RADIUS;
	int j = blockIdx.y * TILE_HEIGHT + threadIdx.y - RADIUS;
	int ind = j * width + i;

	__shared__ float sharedMem[B_W][B_H];
	if (i < 0 || j < 0 || i >= width || j >= height) { 
		sharedMem[threadIdx.x][threadIdx.y] = 0;
		return;
	}

	sharedMem[threadIdx.x][threadIdx.y] = image[ind];

	__syncthreads();

	// box filter (only for threads inside the tile)
	if ((threadIdx.x >= RADIUS) && (threadIdx.x < (B_W - RADIUS)) && (threadIdx.y >= RADIUS) && (threadIdx.y < (B_H - RADIUS))) {
		float sum = 0;
		for (int ix = -RADIUS; ix <= RADIUS; ix++) {
			for (int iy = -RADIUS; iy <= RADIUS; iy++) {
				sum += sharedMem[threadIdx.x + ix][threadIdx.y + iy];
			}
		}
		int val = sum / ((2 * RADIUS + 1)*(2 * RADIUS + 1));
		mean[ind] = val;
		//mean[ind] = 220;
	}
}

__global__ void multIm(unsigned char* im1, unsigned char* im2, float* val, int width, int height) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//N = (4180*2160)
	const int i = index;
	int N = width * height;
	if (i < N)
	{
		val[i] = im1[index] * im2[index];
	}
}

__global__ void sousIm(float* im1, float* im2, float* val, int width, int height) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//N = (4180*2160)
	int i = index;
	int N = width * height;
	if (i < N)
	{
		val[i] = im1[index] - im2[index];
	}
}

void filter(unsigned char* image, int width, int height, unsigned char* mean, float* var, bool cuda)
{
	const int size = 2 * RADIUS + 1;

	int n = width * height;
	memset(mean, 0, n);
	memset(var, 0, sizeof(float)*n);


	unsigned char* d_image;
	unsigned char* d_mean;
	float* d_mean2;
	float* d_var;
	float* d_mult_mean;
	float* d_mult_im;

	cout << "..........." << endl;

	// malloc device global memory
	CHECK(cudaMalloc((unsigned char**)&d_image, n));
	CHECK(cudaMalloc((unsigned char**)&d_mean, n));
	CHECK(cudaMalloc((void**)&d_var, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mult_mean, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mult_im, n * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mean2, n * sizeof(float)));

	CHECK(cudaMemcpy(d_image, image, n, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean, mean, n, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_var, var, sizeof(float)*n, cudaMemcpyHostToDevice));

	dim3 blockDim(B_W, B_H);
	int grid_w = width / TILE_WIDTH + 1;
	int grid_h = height / TILE_HEIGHT + 1;

	dim3 gridDim(grid_w, grid_h);
	boxFilterOnGPU << <gridDim, blockDim >> > (d_image, d_mean, width, height);


	//if (host_gpu_compare) {
	unsigned char* h_mean = (unsigned char*)malloc(n * sizeof(unsigned char));
	memset(h_mean, 0, sizeof(unsigned char)*(n));

	boxFilterOnCPU(image, h_mean, width, height);
	bool verif = check_errors(h_mean, mean, n);
	if (verif) cout << "Cost Volume ok!" << endl;

	free(h_mean);
	//}


	blockDim.x =1024;
	blockDim.y = 1;

	gridDim.x = ((n + blockDim.x - 1) / blockDim.x);
	gridDim.y = 1;
	multIm << <gridDim, blockDim >> > (d_image, d_image, d_mult_im, width, height);
	multIm << <gridDim, blockDim >> > (d_mean, d_mean, d_mult_mean, width, height);

	blockDim.x = B_W;
	blockDim.y = B_H;

	gridDim.x = grid_w;
	gridDim.y = grid_h;
	boxFilterfloatOnGpu << <gridDim, blockDim >> > (d_mult_im, d_mean2, width, height);

	blockDim.x = 1024;
	blockDim.y = 1;
	gridDim.x = ((n + blockDim.x - 1) / blockDim.x);
	gridDim.y = 1;
	sousIm << <gridDim, blockDim >> > (d_mean2, d_mult_mean, d_var, width, height);

	CHECK(cudaDeviceSynchronize());
	//covarOnGpu << <gridDim, blockDim >> > (d_image, d_mean, d_var, width, height);

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(mean, d_mean, n, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(var, d_var, sizeof(float)*n, cudaMemcpyDeviceToHost));

	// free device global memory
	CHECK(cudaFree(d_image));
	CHECK(cudaFree(d_var));
	CHECK(cudaFree(d_mean));
	CHECK(cudaFree(d_mult_im));
	CHECK(cudaFree(d_mean2));
	CHECK(cudaFree(d_mult_mean));

	
}