#include "guidedFilter.cuh"


void compute_guided_filter(unsigned char* i, float* cost, float* filtered ,unsigned char* mean,const int w, const int h, const int size_d, bool host_gpu_compare) {
	int n = w * h;
	int volume = size_d * w*h;
	int n_fl = sizeof(float)*n;
	int radius = 1 * RADIUS;
	//1st step : compute mean filter and its covariance
	// GPU var
	unsigned char* d_i;
	unsigned char* d_mean;
	float* d_im;
	float* d_mean_im;
	float* d_var_im;

	//CPU var

	float* h_im = (float*) malloc(n_fl);
	float* h_mean_im = (float*) malloc (n_fl);
	float* h_var_im = (float*)malloc(n_fl);


	//memset
	memset(mean, 0, n);
	memset(h_im, 0.0f, n_fl);
	memset(h_mean_im, 0.0f, n_fl);
	memset(h_var_im, 0.0f, n_fl);


	//cuda malloc
	CHECK(cudaMalloc((unsigned char**)&d_i, n));
	CHECK(cudaMalloc((unsigned char**)&d_mean, n));
	CHECK(cudaMalloc((void**)&d_im, n_fl));
	CHECK(cudaMalloc((void**)&d_mean_im, n_fl));
	CHECK(cudaMalloc((void**)&d_var_im, n_fl));


	//cuda memcpy host -> device
	CHECK(cudaMemcpy(d_i, i, n, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean, mean, n, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im, h_im, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean_im, h_mean_im, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_var_im, h_var_im, n_fl, cudaMemcpyHostToDevice));


	dim3 blockDim(128);

	dim3 gridDim ((n + blockDim.x - 1) / blockDim.x);
	//im unsigned char -> float
	chToFlOnGPU << <gridDim, blockDim >> > (d_i, d_im, n);

	

	//Compute Integral im1
	CHECK(cudaMemcpy(h_im, d_im, n_fl, cudaMemcpyDeviceToHost));

	float* integral_im = (float*)malloc(n_fl);
	memset(integral_im, 0.0f, n_fl);
	integral(h_im, integral_im, w, h);

	

	float* integral_imCPU = (float*)malloc(n_fl);
	memset(integral_imCPU, 0.0f, n_fl);
	integralOnCPU(h_im, integral_imCPU, w, h);
	check_errors(integral_im, integral_imCPU, w1 * h1);


	/**
	cout << "im(0,0) = " << h_im1[0] << " int(0,0) = " << integral_im1[0] << endl;
	cout << "im(0,1) = " << h_im1[1] << " int(0,1) = " << integral_im1[1] << endl;
	cout << "im(0,2) = " << h_im1[2] << " int(0,2) = " << integral_im1[2] << endl;
	cout << "im(0,3) = " << h_im1[3] << " int(0,3) = " << integral_im1[3] << endl;
	**/

	float* d_integral_im;
	CHECK(cudaMalloc((void**)&d_integral_im, n_fl));
	CHECK(cudaMemcpy(d_integral_im, integral_im, n_fl, cudaMemcpyHostToDevice));
	dim3 y(16, 16);
	dim3 x((w + y.x - 1) / y.x, (h + y.y - 1) / y.y);
	computeBoxFilter << < x, y >> >(d_im, d_integral_im, d_mean_im, (const int) w, (const int) h);
	gridDim.x = (n + blockDim.x -1) / blockDim.x;
	flToChOnGPU << <gridDim, blockDim >> > (d_mean_im, d_mean, n);

	
	//compute variance
	float* d_imSquare;
	float* d_meanSquare;
	float* d_integral_square;
	float* d_temp;
	float* imSquare = (float*)malloc(n_fl);
	float* integral_square = (float*)malloc(n_fl);
	memset(integral_square, 0, n_fl);
	memset(imSquare, 0, n_fl);
	CHECK(cudaMalloc((void**)&d_imSquare, n_fl));
	CHECK(cudaMalloc((void**)&d_meanSquare, n_fl));
	CHECK(cudaMalloc((void**)&d_integral_square, n_fl));
	CHECK(cudaMalloc((void**)&d_temp, n_fl));
	CHECK(cudaMemcpy(d_imSquare, integral_square, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_meanSquare, integral_square, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_integral_square, integral_square, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_temp, integral_square, n_fl, cudaMemcpyHostToDevice));
	dim3 mult(1024);
	dim3 gmult((n + mult.x - 1) / mult.x);
	pixelMultOnGPU << < gmult, mult >> > (d_im, d_im, d_imSquare, n);
	pixelMultOnGPU << < gmult, mult >> > (d_mean_im, d_mean_im, d_meanSquare, n);
	CHECK(cudaMemcpy(imSquare, d_imSquare, n_fl, cudaMemcpyDeviceToHost));
	integral(imSquare, integral_square, w, h);
	CHECK(cudaMemcpy(d_integral_square, integral_square, n_fl, cudaMemcpyHostToDevice));
	computeBoxFilter << < x, y >> >(d_imSquare, d_integral_square, d_temp, (const int)w, (const int)h);
	pixelSousOnGPU << <gridDim, blockDim >> > (d_temp, d_meanSquare, d_var_im,n);
	
	//compute pk, a_k and b_k
	float* d_ak;
	float* d_bk;
	float* d_pk;
	CHECK(cudaMalloc((void**)&d_ak, n_fl));
	CHECK(cudaMalloc((void**)&d_pk, n_fl*size_d));
	CHECK(cudaMalloc((void**)&d_bk, n_fl));
	

	//compute ai, bi and finally qi

















	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());



	CHECK(cudaMemcpy(mean, d_mean, n, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_var_im, d_var_im, n_fl, cudaMemcpyDeviceToHost));

	//free cuda memory
	CHECK(cudaFree(d_i));
	CHECK(cudaFree(d_mean));
	CHECK(cudaFree(d_im));
	CHECK(cudaFree(d_mean_im));
	CHECK(cudaFree(d_var_im));
	CHECK(cudaFree(d_integral_im));
	CHECK(cudaFree(d_integral_square));
	CHECK(cudaFree(d_imSquare));
	CHECK(cudaFree(d_temp));
	CHECK(cudaFree(d_meanSquare));

	//free ram memory
	free(h_im);
	free(h_mean_im);
	free(h_var_im);
	free(integral_im);
	free(integral_square);

}

//CPU functions
__host__ void chToFlOnCPU(unsigned char* image, float* result, int len) {
	for (int i = 0; i < len; i++) {
		unsigned int c = image[i];
		result[i] = 1.0f*c;
	}
}

__host__ void flToChOnCPU(float* image, unsigned char* result, int len) {
	for (int i = 0; i < len; i++)
	{
		unsigned int c = image[i];
		result[i] = (unsigned char)c;
	}
}

__host__ void pixelMultOnCPU(float* image1, float* image2, float* result, int len) {
	for (int i = 0; i < len; i++)
	{
		result[i] = image1[i] * image2[i];
	}
}

__host__ void pixelSousOnCPU(float* image1, float* image2, float* result, int len) {
	for (int i = 0; i < len; i++)
	{
		result[i] = image1[i] - image2[i];
	}
}

__host__ void pixelAddOnCPU(float* image1, float* image2, float* result, int len) {
	for (int i = 0; i < len; i++)
	{
		result[i] = image1[i] + image2[i];
	}
}

__host__ void pixelDivOnCPU(float* image1, float* image2, float* result, int len) {
	for (int i = 0; i < len; i++)
	{
		float c = image2[i];
		if (c != 0) result[i] = image1[i] / c;
	}
}

__global__ void computeBoxFilter(float* image, float* integral, float* mean, const int w, const int h) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx < w && idy < h) {
		mean[idx + w * idy] = computeMean(image, integral, idx, idy, w, h);
	}
}
__device__ float computeMean(float* I, float* S, int idx, int idy, const int w, const int h) {
	int i_x = max(idx - RADIUS, 0);
	int i_y = max(idy - RADIUS, 0);
	int j_x = min((idx + RADIUS), w-1);
	int j_y = min((idy + RADIUS), h-1);
	float S_1 = S[j_y*w + j_x];
	float S_2 = (i_x < 1) ? 0 : S[j_y*w + (i_x - 1)];
	float S_3 = (i_y < 1) ? 0 : S[(i_y - 1)*w + j_x];
	float S_4 = ((i_x < 1) || (i_y < 1)) ? 0 : S[(i_y - 1)*w + (i_x - 1)];
	float area = abs(j_y - i_y)*abs(j_x - i_x);
	return (S_1 + S_4 - S_3 - S_2) / area;
}

/**

// GPU functions

__device__ void mean_x(float *id, float *od, int w, int h, int r)
{
	float scale = 1.0f / (float)((r << 1) + 1);

	float t;
	// do left edge
	t = id[0] * r;

	for (int x = 0; x < (r + 1); x++)
	{
		t += id[x];
	}

	od[0] = t * scale;

	for (int x = 1; x < (r + 1); x++)
	{
		t += id[x + r];
		t -= id[0];
		od[x] = t * scale;
	}

	// main loop
	for (int x = (r + 1); x < w - r; x++)
	{
		t += id[x + r];
		t -= id[x - r - 1];
		od[x] = t * scale;
	}

	// do right edge
	for (int x = w - r; x < w; x++)
	{
		t += id[w - 1];
		t -= id[x - r - 1];
		od[x] = t * scale;
	}
}

__device__ void mean_y(float *id, float *od, int w, int h, int r)
{
	float scale = 1.0f / (float)((r << 1) + 1);

	float t;
	// do left edge
	t = id[0] * r;

	for (int y = 0; y < (r + 1); y++)
	{
		t += id[y * w];
	}

	od[0] = t * scale;

	for (int y = 1; y < (r + 1); y++)
	{
		t += id[(y + r) * w];
		t -= id[0];
		od[y * w] = t * scale;
	}

	// main loop
	for (int y = (r + 1); y < (h - r); y++)
	{
		t += id[(y + r) * w];
		t -= id[((y - r) * w) - w];
		od[y * w] = t * scale;
	}

	// do right edge
	for (int y = h - r; y < h; y++)
	{
		t += id[(h - 1) * w];
		t -= id[((y - r) * w) - w];
		od[y * w] = t * scale;
	}
}

__global__ void compute_mean_x(float *image, float *mean, int w, int h, int radius)
{
	unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
	mean_x(&image[y * w], &mean[y * w], w, h, radius);
}

__global__ void compute_mean_y(float *image, float *mean, int w, int h, int radius)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	mean_y(&image[x], &mean[x], w, h, radius);
}
**/
__global__ void chToFlOnGPU(unsigned char* image, float* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		unsigned int c = image[i];
		result[i] = 1.0f*c / 255;
	}
}

__global__ void flToChOnGPU(float* image, unsigned char* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		int c = image[i] * 255.0;
		result[i] = (c > 255) ? 255 : (unsigned char)c;
	}
}

__global__ void pixelMultOnGPU(float* image1, float* image2, float* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		result[i] = image1[i] * image2[i];
	}
}

__global__ void pixelSousOnGPU(float* image1, float* image2, float* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		result[i] = image1[i] - image2[i];
	}
}

__global__ void pixelAddOnGPU(float* image1, float* image2, float* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		result[i] = image1[i] + image2[i];
	}
}

__global__ void pixelDivOnGPU(float* image1, float* image2, float* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		float c = image2[i];
		if (c != 0) result[i] = image1[i] / c;
	}
}