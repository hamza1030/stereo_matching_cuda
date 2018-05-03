#include "costVolume.cuh"
__host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void compute_cost(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2) {
	int size_d = D_MAX - D_MIN + 1;
	int size_cost = h1 * w1*size_d;
	unsigned char* d_i1;
	unsigned char* d_i2;
	float* h_cost = (float*)malloc(size_cost * sizeof(float));
	float* d_cost;
	memset(cost, 0, sizeof(float)*(size_cost));
	memset(h_cost, 0, sizeof(float)*(size_cost));

	

	CHECK(cudaMalloc((unsigned char**)&d_i1, w1 * h1));
	CHECK(cudaMalloc((unsigned char**)&d_i2, w2 * h2));
	CHECK(cudaMalloc((void**)&d_cost, size_cost * sizeof(float)));
	CHECK(cudaMemcpy(d_i1, i1, w1 * h1, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_i2, i2, w2 * h2, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_cost, cost, sizeof(float)*size_cost, cudaMemcpyHostToDevice));
	dim3 blockDim(32, size_d);
	dim3 gridDim;
	//gridDim.x = (w1*h1 + blockDim.x - 1)/blockDim.x;
	gridDim.x = iDivUp(w1*h1, blockDim.x);
	gridDim.y = 1;//size_d;

	costVolumOnGPU2 << <gridDim, blockDim >> > (d_i1, d_i2, d_cost, w1, w2, h1, h2, size_d);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(cost, d_cost, size_cost * sizeof(float), cudaMemcpyDeviceToHost));

	//host side
	costVolumeOnCPU(i1, i2, h_cost, w1, w2, h1, h2, size_d);
	bool verif = check_errors_cost(h_cost, cost, size_cost);
	if (verif) cout << "Cost Volume ok!" << endl;;

	// free device global memory
	CHECK(cudaFree(d_cost));
	CHECK(cudaFree(d_i1));
	CHECK(cudaFree(d_i2));
	free(h_cost);
}

bool check_errors_cost(float* resCPU, float* resGPU, int len) {
	bool res = true;
	for (int i = 0; i < len; i++) {
		//cout << i << " ResultGPU = " << resGPU[i] << " and ResultCPU= " << resCPU[i] << endl;
		if (resCPU[i] != resGPU[i]) {
			res = false;
			cout << "error at element: " << i << " ResultGPU = " << resGPU[i] << " and ResultCPU= " << resCPU[i] << endl;
		}
	}
	return res;
}

void costVolumeOnCPU(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2, int size_d) {
	float alpha = 1.0f*ALPHA;
	float th_color = 1.0f*TH_color;
	float th_grad = 1.0f*TH_grad;
	for (int d = -D_MIN; d <= D_MAX; d++) {
		for (int y = 0; y < h1; y++) {
			for (int x = 0; x < w1; x++) {
				int index = y * w1 + x;
				int id = d *w1*h1 + index;
				float c = (1.0f - alpha) * th_color + alpha * (1.0f*th_grad);
				if ((x + d < w2) && (x +d >= 0)) {
					float diff_term = 1.0f*abs(i1[index] - i2[index + d]);
					float grad_1 = 1.0f*x_derivativeCPU(i1, x, index, w1);
					float grad_2 = 1.0f*x_derivativeCPU(i2, x + d, index + d, w2);
					float grad_term = 1.0f*abs(grad_1 - grad_2);
					c = (1 - alpha)*min(diff_term, th_color) + alpha * min(grad_term, 1.0f*th_grad);
				}
				cost[id] = c;
			}

		}

	}
}



__global__ void costVolumOnGPU2(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2, int size_d) {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int idx = x % w1;
	int idy = x / w1;
	int id = y*w1*h1 + x;
	int d = -D_MIN + y; //-D_MIN + y;
	if (y < size_d && x < w1*h1) {
		float c = (1.0f - 1.0f*ALPHA) * (1.0f*TH_color) + (1.0f*ALPHA) * (1.0f*TH_grad);
		if (((idx + d) < w2) && ((idx + d) >= 0)) 
		{
			c = (1.0f - 1.0f*ALPHA)*difference_term(i1[x], i2[x + d]) + (1.0f*ALPHA) * difference_term_2(x_derivative(i1, idx, x, w1), x_derivative(i2, idx + d, x + d, w2));
		}
		cost[id] = c;
		//printf("%f\n", c);
	}
}

__device__ int id_im(int i, int j, int width) {
	return j * width + i;
}
__device__ int id_cost(int i, int j, int width, int height, int k) {
	return k * width*height + j * width + i;
}
__device__ float x_derivative(unsigned char* im, int col_index, int index, int width) {
	if ((col_index + 1) < width && (col_index - 1) >= 0)
	{
		return (float) ((im[index + 1] - im[index - 1]) / 2);
	}
	else if (col_index + 1 == width)
	{
		return (float) ((im[index] - im[index - 1]) / 2);
	}
	else
	{
		return (float) ((im[index + 1] - im[index]) / 2);
	}
}

__host__ float x_derivativeCPU(unsigned char* im, int col_index, int index, int width) {
	if ((col_index + 1) < width && (col_index - 1) >= 0)
	{
		return (float)((im[index + 1] - im[index - 1]) / 2);
	}
	else if (col_index + 1 == width)
	{
		return (float)((im[index] - im[index - 1]) / 2);
	}
	else
	{
		return (float)((im[index + 1] - im[index]) / 2);
	}
}

__device__ int difference_term(unsigned char pixel_i, unsigned char pixel_j) {
	return min(abs((int)(pixel_i - pixel_j)), TH_color);
}
__device__ float difference_term_2(float pixel_i, float pixel_j) {
	return min(abs(pixel_i - pixel_j), 1.0f*TH_grad);
}

__device__ int getGlobalIdx_1D_2D()
{
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}