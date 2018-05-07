#include "guidedFilter.cuh"

void compute_guided_filter(unsigned char* i1, unsigned char* i2, float* cost, float* filtered ,unsigned char* mean1, unsigned char* mean2, int w1, int h1, int w2, int h2, int size_d, bool host_gpu_compare) {
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

	dim3 blockDim(128);
	dim3 gridDim ((n1 + blockDim.x - 1) / blockDim.x);
	//im unsigned char -> float
	chToFlOnGPU << <gridDim, blockDim >> > (d_i1, d_im1, n1);
	gridDim.x = (n2 + blockDim.x - 1) / blockDim.x;
	chToFlOnGPU << <gridDim, blockDim >> > (d_i2, d_im2, n2);
	

	//Compute Integral im1
	CHECK(cudaMemcpy(h_im1, d_im1, n1_fl, cudaMemcpyDeviceToHost));
	int im = i1[120];
	float im2 = h_im1[120];
	float* integral_im1 = (float*)malloc(n1_fl);
	memset(integral_im1, 0.0f, n1_fl);
	integral(h_im1, integral_im1, w1, h1);
	/**
	cout << "im(0,0) = " << h_im1[0] << " int(0,0) = " << integral_im1[0] << endl;
	cout << "im(0,1) = " << h_im1[1] << " int(0,1) = " << integral_im1[1] << endl;
	cout << "im(0,2) = " << h_im1[2] << " int(0,2) = " << integral_im1[2] << endl;
	cout << "im(0,3) = " << h_im1[3] << " int(0,3) = " << integral_im1[3] << endl;
	**/
	float* d_integral_im1;
	CHECK(cudaMalloc((void**)&d_integral_im1, n1_fl));
	CHECK(cudaMemcpy(d_integral_im1, integral_im1, n1_fl, cudaMemcpyHostToDevice));
	dim3 y1(16, 16);
	dim3 x1((w1 + y1.x - 1) / y1.x, (h1 + y1.y - 1) / y1.y);
	computeBoxFilter << < x1, y1 >> >(d_im1, d_integral_im1, d_mean_im1, (const int) w1, (const int) h1);
	gridDim.x = (n1 + blockDim.x -1) / blockDim.x;
	flToChOnGPU << <gridDim, blockDim >> > (d_mean_im1, d_mean1, n1);


	//compute intgral im2
	CHECK(cudaMemcpy(h_im2, d_im2, n2_fl, cudaMemcpyDeviceToHost));
	float* integral_im2 = (float*)malloc(n2_fl);
	memset(integral_im2, 0.0f, n2_fl);
	integral(h_im2, integral_im2, w2, h2);
	float* d_integral_im2;
	CHECK(cudaMalloc((void**)&d_integral_im2, n2_fl));
	CHECK(cudaMemcpy(d_integral_im2, integral_im2, n2_fl, cudaMemcpyHostToDevice));
	dim3 y2(16,16);
	dim3 x2((w2 + y2.x - 1) / y2.x, (h2 + y2.y - 1) / y2.y);
	computeBoxFilter << < x2, y2 >> >(d_im2, d_integral_im2, d_mean_im2, (const int)w2, (const int)h2);
	gridDim.x = (n2 + blockDim.x - 1) / blockDim.x;
	flToChOnGPU << <gridDim, blockDim >> > (d_mean_im2, d_mean2, n2);
	
	
	//compute mean im1
	

	//compute mean im2

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());


	CHECK(cudaMemcpy(mean1, d_mean1, n1, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(mean2, d_mean2, n2, cudaMemcpyDeviceToHost));
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
	CHECK(cudaFree(d_integral_im1));
	CHECK(cudaFree(d_integral_im2));


	//free ram memory
	free(h_im1);
	free(h_im2);
	free(h_mean_im1);
	free(h_mean_im2);
	free(h_var_im1);
	free(h_var_im2);
	free(integral_im1);
	free(integral_im2);
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







__global__ void computeBoxFilter(float* image, float* integral, float* mean, const int w, const int h) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx < w && idy<h) {
		mean[idx + w * idy] = computeMean(image, integral, idx, idy, w, h);
	}

}
__device__ float computeMean(float* I, float* S, int idx, int idy, const int w, const int h) {
	int i_x = max(idx - RADIUS, 0);
	int i_y = max(idy - RADIUS, 0);
	int j_x = min((idx + RADIUS + 1), w - 1);
	int j_y = min((idy + RADIUS + 1), h - 1);
	float S_1 = S[j_y*w + j_x];
	float S_2 = (i_x<1) ? 0 : S[j_y*w + (i_x - 1)];
	float S_3 = (i_y<1) ? 0 : S[(i_y - 1)*w + j_x];
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
		result[i] = 1.0f*c/255;
	}

}

__global__ void flToChOnGPU(float* image, unsigned char* result, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		int c = image[i]*255.0;
		result[i] = (c>255)?255: (unsigned char) c;
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