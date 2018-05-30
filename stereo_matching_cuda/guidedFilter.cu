#include "guidedFilter.cuh"

void compute_guided_filter(unsigned char* i, float* cost, float* filter_cost, float* disp_map, unsigned char* mean, const int w, const int h, const int size_d, int dmin, bool host_gpu_compare) {
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
	float* d_cost;
	float* d_filt_cost;
	float* d_dmap;

	//CPU var
	float* fMean = (float*)malloc(n_fl);

	float* h_im = (float*)malloc(n_fl);
	float* h_mean_im = (float*)malloc(n_fl);
	float* h_var_im = (float*)malloc(n_fl);
	float* empty = (float*)malloc(n_fl);
	float* big_empty = (float*)malloc(size_d*n_fl);

	//memset
	memset(fMean, 0, n);
	memset(mean, 0, n);
	memset(h_im, 0.0f, n_fl);
	memset(h_mean_im, 0.0f, n_fl);
	memset(h_var_im, 0.0f, n_fl);
	memset(empty, 0.0f, n_fl);
	memset(big_empty, 0.0f, n_fl*size_d);

	//cuda malloc
	CHECK(cudaMalloc((unsigned char**)&d_i, n));
	CHECK(cudaMalloc((unsigned char**)&d_mean, n));
	CHECK(cudaMalloc((void**)&d_im, n_fl));
	CHECK(cudaMalloc((void**)&d_mean_im, n_fl));
	CHECK(cudaMalloc((void**)&d_var_im, n_fl));
	CHECK(cudaMalloc((void**)&d_cost, size_d*n_fl));
	CHECK(cudaMalloc((void**)&d_filt_cost, n_fl));
	CHECK(cudaMalloc((void**)&d_dmap, n_fl));

	//cuda memcpy host -> device
	CHECK(cudaMemcpy(d_i, i, n, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean, mean, n, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im, h_im, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_mean_im, h_mean_im, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_filt_cost, filter_cost, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_dmap, disp_map, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_cost, cost, size_d*n_fl, cudaMemcpyHostToDevice));

	float* h_filt_cost = (float*)malloc(n_fl * sizeof(float));
	memset(h_filt_cost, 0, sizeof(float)*(n_fl));

	CHECK(cudaMemcpy(h_filt_cost, d_filt_cost, n_fl, cudaMemcpyDeviceToHost));

	dim3 blockDim(128);

	dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
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
	check_errors(integral_im, integral_imCPU, w * h);


	float* d_integral_im;
	CHECK(cudaMalloc((void**)&d_integral_im, n_fl));
	CHECK(cudaMemcpy(d_integral_im, integral_im, n_fl, cudaMemcpyHostToDevice));
	dim3 y(16, 16);
	dim3 x((w + y.x - 1) / y.x, (h + y.y - 1) / y.y);
	computeBoxFilterOnGPU << < x, y >> > (d_im, d_integral_im, d_mean_im, (const int)w, (const int)h);

	CHECK(cudaMemcpy(fMean, d_mean_im, n, cudaMemcpyDeviceToHost));
	

	computeBoxFilterOnCPU(h_im, integral_imCPU, h_mean_im, (const int)w, (const int)h);
	//for (size_t i = 0; i < n; i++)
	//{
	//	cout << fMean[i] << endl;
	//}
	bool verif = check_errors(h_mean_im, fMean, n);
	if (verif) cout << "Box filter ok!" << endl;


	gridDim.x = (n + blockDim.x - 1) / blockDim.x;
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
	CHECK(cudaMemcpy(d_imSquare, empty, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_meanSquare, empty, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_integral_square, empty, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_temp, empty, n_fl, cudaMemcpyHostToDevice));
	dim3 mult(1024);
	dim3 gmult((n + mult.x - 1) / mult.x);

	// I*I AND mean*mean
	pixelMultOnGPU << < gmult, mult >> > (d_im, d_im, d_imSquare, n);
	pixelMultOnGPU << < gmult, mult >> > (d_mean_im, d_mean_im, d_meanSquare, n);
	CHECK(cudaMemcpy(imSquare, d_imSquare, n_fl, cudaMemcpyDeviceToHost));

	//mean(I*I)
	integral(imSquare, integral_square, w, h);
	CHECK(cudaMemcpy(d_integral_square, integral_square, n_fl, cudaMemcpyHostToDevice));
	computeBoxFilterOnGPU << < x, y >> > (d_imSquare, d_integral_square, d_temp, (const int)w, (const int)h);

	//var = mean(I*I) - mean*mean
	pixelSousOnGPU << <gridDim, blockDim >> > (d_temp, d_meanSquare, d_var_im, n);
	CHECK(cudaMemcpy(mean, d_mean, n, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_var_im, d_var_im, n_fl, cudaMemcpyDeviceToHost));

	//compute pk, a_k and b_k, a_i, b_i, q

	//1. variable in device
	float* d_ak;
	float* d_bk;
	float* d_ak_int;
	float* d_bk_int;
	float* d_ak_mean;
	float* d_bk_mean;
	float* d_pk;
	float* d_pki;
	float* d_pki_mean;
	float* d_convol;
	float* d_convol_int;
	float* d_q;
	float* d_convol_mean;
	float* d_pki_int;
	//variable on cpu
	float* pki_int = (float*)malloc(n_fl);
	float* pki = (float*)malloc(n_fl);
	float* convol = (float*)malloc(n_fl);
	float* convol_int = (float*)malloc(n_fl);
	float* ak = (float*)malloc(n_fl);
	float* bk = (float*)malloc(n_fl);
	float* ak_int = (float*)malloc(n_fl);
	float* bk_int = (float*)malloc(n_fl);

	CHECK(cudaMalloc((void**)&d_pki, n_fl*size_d));
	CHECK(cudaMalloc((void**)&d_pki_int, n_fl));
	CHECK(cudaMalloc((void**)&d_pki_mean, n_fl));
	CHECK(cudaMalloc((void**)&d_ak, n_fl));
	CHECK(cudaMalloc((void**)&d_bk, n_fl));
	CHECK(cudaMalloc((void**)&d_ak_int, n_fl));
	CHECK(cudaMalloc((void**)&d_bk_int, n_fl));
	CHECK(cudaMalloc((void**)&d_ak_mean, n_fl));
	CHECK(cudaMalloc((void**)&d_bk_mean, n_fl));
	CHECK(cudaMalloc((void**)&d_convol, n_fl));
	CHECK(cudaMalloc((void**)&d_convol_int, n_fl));
	CHECK(cudaMalloc((void**)&d_convol_mean, n_fl));
	CHECK(cudaMalloc((void**)&d_q, n_fl));
	dim3 bdim(1024);
	dim3 gdim((n + bdim.x - 1) / bdim.x);
	dim3 bdim2(16, 16);
	dim3 gdim2((w + bdim2.x - 1) / bdim2.x, (h + bdim2.y - 1) / bdim2.y);

	//loop over d range
	for (int s = 0; s < size_d; s++) {
		int start = s * n;
		memset(pki_int, 0.0f, n_fl);
		memset(pki, 0.0f, n_fl);
		memset(convol, 0.0f, n_fl);
		memset(convol_int, 0.0f, n_fl);
		memset(ak, 0.0f, n_fl);
		memset(ak_int, 0.0f, n_fl);
		memset(bk, 0.0f, n_fl);
		memset(bk_int, 0.0f, n_fl);
		CHECK(cudaMemcpy(d_pki, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pki_int, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pki_mean, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_convol, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_convol_int, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_convol_mean, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_ak, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_bk, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_ak_int, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_bk_int, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_ak_mean, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_bk_mean, empty, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_q, empty, n_fl, cudaMemcpyHostToDevice));

		//Cost -> pk
		copyFromBigToLittleOnGPU << <gdim, bdim >> > (d_cost, d_pki, start, n);
		CHECK(cudaMemcpy(pki, d_pki, n_fl, cudaMemcpyDeviceToHost));

		//for (int i = 0; i < 10; i++) { cout << pki[i] << " =? " << cost[i] <<endl; }

		//compute pk_mean
		integral(pki, pki_int, w, h);
		CHECK(cudaMemcpy(d_pki_int, pki_int, n_fl, cudaMemcpyHostToDevice));
		computeBoxFilterOnGPU << < gdim2, bdim2 >> > (d_pki, d_pki_int, d_pki_mean, (const int)w, (const int)h);

		//I*p
		pixelMultOnGPU << <gdim, bdim >> > (d_im, d_pki, d_convol, n);
		CHECK(cudaMemcpy(convol, d_convol, n_fl, cudaMemcpyDeviceToHost));

		//mean(I*p)
		integral(convol, convol_int, w, h);
		CHECK(cudaMemcpy(d_convol_int, convol_int, n_fl, cudaMemcpyHostToDevice));
		computeBoxFilterOnGPU << < gdim2, bdim2 >> > (d_convol, d_convol_int, d_convol_mean, (const int)w, (const int)h);

		//Compute ak and bk
		compute_ak_and_bk << <gdim, bdim >> > (d_mean_im, d_var_im, d_convol_mean, d_pki_mean, d_ak, d_bk, n);

		//compute ai, bi
		CHECK(cudaMemcpy(ak, d_ak, n_fl, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(bk, d_bk, n_fl, cudaMemcpyDeviceToHost));
		integral(ak, ak_int, w, h);
		integral(bk, bk_int, w, h);
		CHECK(cudaMemcpy(d_ak_int, ak_int, n_fl, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_bk_int, bk_int, n_fl, cudaMemcpyHostToDevice));
		computeBoxFilterOnGPU << < gdim2, bdim2 >> > (d_ak, d_ak_int, d_ak_mean, (const int)w, (const int)h);
		computeBoxFilterOnGPU << < gdim2, bdim2 >> > (d_bk, d_bk_int, d_bk_mean, (const int)w, (const int)h);

		//compute qi
		compute_q << <gdim, bdim >> > (d_im, d_ak_mean, d_bk_mean, d_q, n);
		//int label = dmin + s;
		int label = dmin + s;
		//qi ->total filtered
		dispSelectOnGPU << <gdim, bdim >> > (d_q, d_filt_cost, d_dmap, (const int)n, label);
	}

	float* h_q = (float*)malloc(n_fl * sizeof(float));
	memset(h_q, 0, sizeof(float)*(n_fl));

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(filter_cost, d_filt_cost, n_fl, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(disp_map, d_dmap, n_fl, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(h_q, d_q, n_fl, cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < n; i++)
	{
		cout << filter_cost[i] << endl;
	}

	//for (int s = 0; s < size_d; s++)
	//{
	//	int label = dmin + s;

	//	dispSelectOnCPU(h_q, h_filt_cost, disp_map, (const int)n, label);
	//}

	//

	//bool verif = check_errors(h_filt_cost, filter_cost, n_fl);
	//if (verif) cout << "Disp select ok!" << endl;

	//free(h_filt_cost);
	//free(h_q);

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
	CHECK(cudaFree(d_pki));
	CHECK(cudaFree(d_ak));
	CHECK(cudaFree(d_bk));
	CHECK(cudaFree(d_ak_int));
	CHECK(cudaFree(d_bk_int));
	CHECK(cudaFree(d_ak_mean));
	CHECK(cudaFree(d_bk_mean));
	CHECK(cudaFree(d_pki_int));
	CHECK(cudaFree(d_pki_mean));
	CHECK(cudaFree(d_cost));
	CHECK(cudaFree(d_filt_cost));
	CHECK(cudaFree(d_dmap));
	CHECK(cudaFree(d_convol_int));
	CHECK(cudaFree(d_convol));
	CHECK(cudaFree(d_convol_mean));
	CHECK(cudaFree(d_q));

	//free ram memory
	free(h_im);
	free(h_mean_im);
	free(h_var_im);
	free(integral_im);
	free(integral_square);
	free(empty);
	free(big_empty);
	free(pki);
	free(pki_int);
	free(convol);
	free(convol_int);
	free(ak);
	free(ak_int);
	free(bk);
	free(bk_int);
}


__global__ void computeBoxFilterOnGPU(float* image, float* integral, float* mean, const int w, const int h) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	if (idx < w && idy < h) {
		mean[idx + w * idy] = computeMeanOnGPU(image, integral, idx, idy, w, h);
	}
}
__device__ float computeMeanOnGPU(float* I, float* S, int idx, int idy, const int w, const int h) {
	/**
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
	**/
	int ymin = max(-1, idy - RADIUS - 1);
	int ymax = min(h - 1, idy + RADIUS);
	int xmin = max(-1, idx - RADIUS - 1);
	int xmax = min(w - 1, idx + RADIUS);
	float val = S[ymax*w + xmax];
	if (xmin >= 0)
		val -= S[ymax*w + xmin];
	if (ymin >= 0)
		val -= S[ymin*w + xmax];
	if (xmin >= 0 && ymin >= 0)
		val += S[ymin*w + xmin];
	return (1.0f*val / ((xmax - xmin)*(ymax - ymin)));
}

void computeBoxFilterOnCPU(float* image, float* integral, float* mean, const int w, const int h)
{
	for (size_t i = 0; i < w; i++)
	{
		for (size_t j = 0; j < h; j++)
		{
			mean[i + w * j] = computeMeanOnCPU(image, integral, i, j, w, h);
		}
	}
}
float computeMeanOnCPU(float* I, float* S, int idx, int idy, const int w, const int h)
{
	int ymin = max(-1, idy - RADIUS - 1);
	int ymax = min(h - 1, idy + RADIUS);
	int xmin = max(-1, idx - RADIUS - 1);
	int xmax = min(w - 1, idx + RADIUS);
	float val = S[ymax*w + xmax];
	if (xmin >= 0)
		val -= S[ymax*w + xmin];
	if (ymin >= 0)
		val -= S[ymin*w + xmax];
	if (xmin >= 0 && ymin >= 0)
		val += S[ymin*w + xmin];
	return (1.0f*val / ((xmax - xmin)*(ymax - ymin)));
}

// q calculations

__global__ void compute_ak(float* mean, float* var, float* convol, float* pk, float* a, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float c;
	if (i < len)
	{
		c = 1.0f / (var[i] + EPS);
		a[i] = 1.0f*(convol[i] - mean[i] * pk[i]) / c;
	}
}

__global__ void compute_ak_and_bk(float* mean, float* var, float* convol, float* pk, float* a, float* b, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float c;
	if (i < len)
	{
		c = 1.0f / (var[i] + EPS);
		a[i] = 1.0f*(convol[i] - mean[i] * pk[i]) / c;
		b[i] = 1.0f*pk[i] - 1.0f*mean[i] * a[i];
	}
}

__global__ void compute_bk(float* mean, float* a, float* pk, float* b, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		b[i] = 1.0f*pk[i] - 1.0f*mean[i] * a[i];
	}
}
__global__ void compute_q(float* im, float* a, float* b, float* q, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		q[i] = a[i] * im[i] + b[i];
	}
}

// CPU
void compute_akOnCPU(float* mean, float* var, float* convol, float* pk, float* a, int len) {
	for (size_t i = 0; i < len; i++)
	{
		float c = 1.0f / (var[i] + EPS);
		a[i] = 1.0f*(convol[i] - mean[i] * pk[i]) / c;
	}
}

void compute_ak_and_bkOnCPU(float* mean, float* var, float* convol, float* pk, float* a, float* b, int len) {
	for (size_t i = 0; i < len; i++)
	{
		float c = 1.0f / (var[i] + EPS);
		a[i] = 1.0f*(convol[i] - mean[i] * pk[i]) / c;
		b[i] = 1.0f*pk[i] - 1.0f*mean[i] * a[i];
	}
}

void compute_bkOnCPU(float* mean, float* a, float* pk, float* b, int len) {
	for (size_t i = 0; i < len; i++)
	{
		b[i] = 1.0f*pk[i] - 1.0f*mean[i] * a[i];
	}
}
void compute_qOnCPU(float* im, float* a, float* b, float* q, int len) {
	for (size_t i = 0; i < len; i++)
	{
		q[i] = a[i] * im[i] + b[i];
	}
}

//disp selection
__global__ void dispSelectOnGPU(float* q, float* filter_cost, float* dmap, const int n, int label) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id < n) {
		if (1.0f*filter_cost[id] >= 1.0f*q[id]) {
			dmap[id] = label;
			filter_cost[id] = q[id];
		}
	}
}

void dispSelectOnCPU(float* q, float* filter_cost, float* dmap, const int n, int label)
{
	for (size_t i = 0; i < n; i++)
	{
		if (1.0f*filter_cost[i] >= 1.0f*q[i]) {
			dmap[i] = label;
			filter_cost[i] = q[i];
		}
	}
}


// simple operations"
// GPU
__global__ void copyFromBigToLittleOnGPU(float* image1, float* result, int start, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		result[i] = image1[start + i];
	}
}
__global__ void copyFromLittleToBigOnGPU(float* image1, float* result, int start, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		result[i + start] = image1[i];
	}
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
		int c = image[i];
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