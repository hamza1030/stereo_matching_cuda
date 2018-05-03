#include "guidedFilter.cuh"

void compute_guided_filter(unsigned char* i1, unsigned char* i2, float* cost, unsigned char* mean1, unsigned char* mean2, int w1, int h1, int w2, int h2, int size_d, bool host_gpu_compare) {
	int n1 = w1 * h1;
	int n2 = w2 * h2;
	int volume = size_d * w1*h1;
	int n1_fl = sizeof(float)*n1;
	int n2_fl = sizeof(float)*n2;
	int radius = 1 * RADIUS;
 	//1st step : compute mean filter and its covariance
	// GPU var
	unsigned char* d_i1;
	unsigned char* d_i2;
	unsigned char* d_mean1;
	unsigned char* d_mean2;
	float* d_im1;
	float* d_im2;
	float* d_mean_im1;
	float* d_mean_im2;
	float* d_var_im1;
	float* d_var_im2;

	//CPU var
	float* h_im1 = (float*) malloc(n1_fl);
	float* h_im2 = (float*) malloc(n2_fl);
	float* h_mean_im1 = (float*) malloc (n1_fl);
	float* h_mean_im2 = (float*) malloc (n2_fl);
	float* h_var_im1 = (float*)malloc(n1_fl);
	float* h_var_im2 = (float*)malloc(n2_fl);

	//memset
	memset(mean1, 0, n1);
	memset(mean2, 0, n2);
	memset(h_im1, 0.0f, n1_fl);
	memset(h_im2, 0.0f, n2_fl);
	memset(h_mean_im1, 0.0f, n1_fl);
	memset(h_mean_im2, 0.0f, n2_fl);
	memset(h_var_im1, 0.0f, n1_fl);
	memset(h_var_im2, 0.0f, n2_fl);

	//cuda malloc
	CHECK(cudaMalloc((unsigned char**)&d_i1, n1));
	CHECK(cudaMalloc((unsigned char**)&d_i2, n2));
	CHECK(cudaMalloc((unsigned char**)&d_mean1, n1));
	CHECK(cudaMalloc((unsigned char**)&d_mean2, n2));
	CHECK(cudaMalloc((void**)&d_im1, n1_fl));
	CHECK(cudaMalloc((void**)&d_im2, n2_fl));
	CHECK(cudaMalloc((void**)&d_mean_im1, n1_fl));
	CHECK(cudaMalloc((void**)&d_mean_im2, n2_fl));
	CHECK(cudaMalloc((void**)&d_var_im1, n1_fl));
	CHECK(cudaMalloc((void**)&d_var_im2, n2_fl));

	//cuda memcpy host -> device
	CHECK(cudaMemcpy(d_i1, i1, n1, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_i2, i2, n2, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean1, mean1, n1, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean2, mean2, n2, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im1, h_im1, n1_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im2, h_im2, n2_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean_im1, h_mean_im1, n1_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean_im2, h_mean_im2, n2_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_var_im1, h_var_im1, n1_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_var_im2, h_var_im2, n2_fl, cudaMemcpyHostToDevice));

	dim3 blockDim(1024);
	dim3 gridDim((blockDim.x + n1 - 1) / blockDim.x);
	//im unsigned char -> float
	chToFlOnGPU << <gridDim, blockDim >> > (d_i1, d_im1, n1);
	chToFlOnGPU << <gridDim, blockDim >> > (d_i1, d_mean_im1, n1);
	gridDim.x = (blockDim.x + n2 - 1) / blockDim.x;
	chToFlOnGPU << <gridDim, blockDim >> > (d_i2, d_im2, n2);
	chToFlOnGPU << <gridDim, blockDim >> > (d_i1, d_mean_im2, n2);

	//compute mean im1
	float* temp_im1;
	float* empty = (float*)malloc(n1_fl);
	CHECK(cudaMalloc((void**)&temp_im1, n1_fl));
	CHECK(cudaMemcpy(temp_im1, empty, n1_fl, cudaMemcpyHostToDevice));
	gridDim.x = (h1 + blockDim.x - 1) / blockDim.x;
	compute_mean_x << <gridDim, blockDim >> > (d_mean_im1, temp_im1, w1, h1, radius);
	gridDim.x = (w1 + blockDim.x - 1) / blockDim.x;
	compute_mean_y << <gridDim, blockDim >> > (temp_im1, d_mean_im1 ,w1, h1, radius);
	free(empty);
	CHECK(cudaFree(temp_im1));
	gridDim.x = (blockDim.x + n1 - 1) / blockDim.x;
	flToChOnGPU << <gridDim, blockDim >> > (d_mean_im1, d_mean1, n1);


	//compute mean im2
	float* temp_im2;
	float* empty2 = (float*)malloc(n2_fl);
	CHECK(cudaMalloc((void**)&temp_im2, n2_fl));
	CHECK(cudaMemcpy(temp_im2, empty2, n2_fl, cudaMemcpyHostToDevice));
	gridDim.x = (h2 + blockDim.x - 1) / blockDim.x;
	compute_mean_x << <gridDim, blockDim >> > (d_mean_im2, temp_im2, w2, h2, radius);
	gridDim.x = (w2 + blockDim.x - 1) / blockDim.x;
	compute_mean_y << <gridDim, blockDim >> > (temp_im2, d_mean_im2, w2, h2, radius);
	free(empty2);
	CHECK(cudaFree(temp_im2));
	gridDim.x = (blockDim.x + n2 - 1) / blockDim.x;
	flToChOnGPU << <gridDim, blockDim >> > (d_mean_im2, d_mean2, n2);



	//free cuda memory
	CHECK(cudaFree(d_i1));
	CHECK(cudaFree(d_i2));
	CHECK(cudaFree(d_mean1));
	CHECK(cudaFree(d_mean2));
	CHECK(cudaFree(d_im1));
	CHECK(cudaFree(d_im2));
	CHECK(cudaFree(d_mean_im1));
	CHECK(cudaFree(d_mean_im2));
	CHECK(cudaFree(d_var_im1));
	CHECK(cudaFree(d_var_im2));

	//free ram memory
	free(h_im1);
	free(h_im2);
	free(h_mean_im1);
	free(h_mean_im2);
	free(h_var_im1);
	free(h_var_im2);
}

//CPU functions
__host__ void chToFlOnCPU(unsigned char* image, float* result, int len) {
	for (int i = 0; i < len; i++) {
		unsigned int c = image[i];
		result[i] = 1.0f*c;

	}

}

__host__ void flToChOnCPU(float* image, unsigned char* result, int len) {
	for  (int i = 0; i< len; i++)
	{
		unsigned int c = image[i];
		result[i] = (unsigned char)c;
	}

}

__host__ void pixelMultOnCPU(float* image1, float* image2, float* result, int len) {
	for (int i = 0; i< len; i++)
	{
		result[i] = image1[i] * image2[i];
	}

}

__host__ void pixelSousOnCPU(float* image1, float* image2, float* result, int len) {
	for (int i = 0; i< len; i++)
	{
		result[i] = image1[i] - image2[i];
	}

}

__host__ void pixelAddOnCPU(float* image1, float* image2, float* result, int len) {
	for (int i = 0; i< len; i++)
	{
		result[i] = image1[i] + image2[i];
	}

}

__host__ void pixelDivOnCPU(float* image1, float* image2, float* result, int len) {

	for (int i = 0; i< len; i++)
	{
		float c = image2[i];
		if (c != 0) result[i] = image1[i] / c;
	}

}












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

__global__ void chToFlOnGPU(unsigned char* image, float* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		unsigned int c = image[i];
		result[i] = 1.0f*c;
	}

}

__global__ void flToChOnGPU(float* image, unsigned char* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		unsigned int c = image[i];
		result[i] = (unsigned char) c;
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
		if (c != 0) result[i] = image1[i] /c;
	}

}