#include "costVolume.cuh"
__host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void compute_cost(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2, int dmin, bool host_gpu_compare) {
	int size_d = D_MAX - D_MIN + 1;
	int size_cost = h1 * w1*size_d;
	unsigned char* d_i1;
	unsigned char* d_i2;
	float* d_xder1;
	float* d_xder2;
	float* d_cost;
	float* derivative1 = (float*)malloc(h1*w1 * sizeof(float));
	float* derivative2 = (float*)malloc(h2*w2 * sizeof(float));
	memset(derivative1, 0.0f, h1*w1 * sizeof(float));
	memset(derivative2, 0.0f, h2*w2 * sizeof(float));
	memset(cost, 0.0f, size_d*h2*w2 * sizeof(float));
	

	CHECK(cudaMalloc((unsigned char**)&d_i1, w1 * h1));
	CHECK(cudaMalloc((unsigned char**)&d_i2, w2 * h2));
	CHECK(cudaMalloc((void**)&d_xder1, w1 * h1 * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_xder2, w2 * h2 * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_cost, size_cost * sizeof(float)));
	CHECK(cudaMemcpy(d_i1, i1, w1 * h1, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_i2, i2, w2 * h2, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_xder1, derivative1, w1 * h1*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_xder2, derivative2, w2 * h2*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_cost, cost, sizeof(float)*size_cost, cudaMemcpyHostToDevice));
	dim3 blockDim1(1024);
	dim3 gridDim1 ((h1*w1 + blockDim1.x -1)/blockDim1.x);
	dim3 blockDim2(1024);
	dim3 gridDim2((h2*w2 + blockDim2.x - 1) / blockDim2.x);
	x_derivativeOnGPU << <gridDim1, blockDim1 >> >(d_i1, d_xder1, w1, h1);
	x_derivativeOnGPU << <gridDim2, blockDim2 >> >(d_i2, d_xder2, w2, h2);
	CHECK(cudaMemcpy(derivative1, d_xder1, h1*w1 * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(derivative2, d_xder2, h2*w2 * sizeof(float), cudaMemcpyDeviceToHost));

	dim3 blockDim(32, size_d);
	dim3 gridDim;
	//gridDim.x = (w1*h1 + blockDim.x - 1)/blockDim.x;
	gridDim.x = iDivUp(w1*h1, blockDim.x);
	gridDim.y = 1;//size_d;

	costVolumOnGPU2 << <gridDim, blockDim >> > (d_i1, d_i2, d_cost, d_xder1, d_xder2, w1, w2, h1, h2, size_d, dmin);


	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(cost, d_cost, size_cost * sizeof(float), cudaMemcpyDeviceToHost));

	//host side
	if (host_gpu_compare) {
		float* h_cost = (float*)malloc(size_cost * sizeof(float));
		memset(h_cost, 0, sizeof(float)*(size_cost));

		costVolumeOnCPU(i1, i2, h_cost, w1, w2, h1, h2, size_d, dmin);
		bool verif = check_errors(h_cost, cost, size_cost);
		if (verif) cout << "Cost Volume ok!" << endl;
		//costVolumeOnCPU(i1, i2, cost, w1, w2, h1, h2, size_d, dmin);
		
		free(h_cost);
	}

	// free device global memory
	CHECK(cudaFree(d_cost));
	CHECK(cudaFree(d_i1));
	CHECK(cudaFree(d_i2));
	CHECK(cudaFree(d_xder1));
	CHECK(cudaFree(d_xder2));
	free(derivative1);
	free(derivative2);
}
void disparity_selection(float* filtered_cost, float* best_cost, float* disparity_map, const int w, const int h, const int dmin,bool host_gpu_compare) {
	const int size_d = D_MAX - D_MIN + 1;
	const int n = w * h;
	int n_fl = n * sizeof(float);
	float* d_filtered_cost;
	float* d_best_cost;
	float* d_dmap;
	CHECK(cudaMalloc((void**)&d_best_cost, n_fl));
	CHECK(cudaMalloc((void**)&d_filtered_cost, size_d*n_fl));
	CHECK(cudaMalloc((void**)&d_dmap, n_fl));
	CHECK(cudaMemcpy(d_filtered_cost, filtered_cost, size_d*n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_dmap, disparity_map, n_fl, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_best_cost, best_cost, n_fl, cudaMemcpyHostToDevice));

	dim3 blockDim(1024);
	dim3 gridDim((n +blockDim.x -1)/blockDim.x);
	//gridDim.x = (w1*h1 + blockDim.x - 1)/blockDim.x;

	selectionOnGpu<< <gridDim, blockDim >> > (d_filtered_cost, d_best_cost, d_dmap, n, size_d, dmin);



	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(best_cost, d_best_cost, n_fl, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(disparity_map, d_dmap, n_fl, cudaMemcpyDeviceToHost));


	// free device global memory
	CHECK(cudaFree(d_best_cost));
	CHECK(cudaFree(d_dmap));
	CHECK(cudaFree(d_filtered_cost));
}

__global__ void selectionOnGpu(float* filt_cost, float* best_cost, float* dmap, const int n, const int dsize, const int dmin) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int offset = n;
	if (i < n) {
		for (int j = 0; j < dsize; j++) {
			if (1.0f*best_cost[i] > 1.0f*filt_cost[i + j * n]) {
				best_cost[i] = filt_cost[i + j * n];
				dmap[i] =  dmin + j;
			}
		}
	}
	
}



void costVolumeOnCPU(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2, int size_d, int dmin) {
	float alpha = 1.0f*ALPHA;
	float th_color = 1.0f*TH_color;
	float th_grad = 1.0f*TH_grad;
	for (int z = 0; z < size_d; z++) {
		for (int y = 0; y < h1; y++) {
			for (int x = 0; x < w1; x++) {
				int index = y * w1 + x;
				int id = z * w1*h1 + index;
				float c = (1.0f - alpha) * th_color + alpha * (1.0f*th_grad);
				int d = dmin + z;
				if ((x + d < w2) && (x + d >= 0)) {
					float diff_term = 1.0f*abs(i1[index] - i2[index + d]);
					float grad_1 = x_derivativeCPU(i1, x, index, w1);
					float grad_2 = x_derivativeCPU(i2, x + d, index + d, w2);
					float grad_term = abs(grad_1 - grad_2);
					c = (1 - alpha)*min(diff_term, th_color) + alpha * min(grad_term, th_grad);
				}
				cost[id] = c;
			}
		}
	}
}

__global__ void costVolumOnGPU2(unsigned char* i1, unsigned char* i2, float* cost, float* derivative1,float* derivative2,int w1, int w2, int h1, int h2, int size_d, int dmin) {
	// x threads for pixels [0, w*h]
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	// y threads for d [0, size_d]
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	float alpha = 1.0f*ALPHA;
	float th_color = 1.0f*TH_color;
	float th_grad = 1.0f*TH_grad;

	// row index in the image
	int idx = x % w1;
	// col index in the image
	int idy = x / w1;
	// index [0, w*h*size_d]
	int id = y * w1*h1 + x;
	// d candidate [dmin, dmax]
	int d = dmin + y;

	if (y < size_d && x < w1*h1) {
		// threshold
		float c = (1 - alpha) * th_color + alpha * th_grad;
		if (((idx + d) < w2) && ((idx + d) >= 0)) 
		{
			c = (1 - alpha)*difference_term(i1[x], i2[x + d]) + alpha * difference_term_2(derivative1[x], derivative2[x + d]);
		}
		cost[id] = c;
		//printf("%f\n", c);

		//float* q;
		//// TODO filter
		//q[id] = 0;

		//__syncthreads();

		//// disparity selection - blockDim should be SIZE_1D !!!

		//// fill with 0
		//__shared__ float bestDisparity[SIZE_1D];
		//// fill with 100000
		//__shared__ float bestCost[SIZE_1D];

		//bestDisparity[threadIdx.x] = 0;
		//bestCost[threadIdx.x] = 0;

		//__syncthreads();

		//if (q[id] < bestCost[threadIdx.x])
		//{
		//	bestCost[threadIdx.x] = q[id];
		//	bestDisparity[threadIdx.x] = d;
		//}

		//__syncthreads();

		//// output to add in param - size w*h - fill with 0
		//float* disparityMap;
		//disparityMap[x] = bestDisparity[threadIdx.x];
	}

	//extern __shared__ float temp[];
	//// for shared memory
	//int tdx = threadIdx.x;
	//// to cumSum one row - for w = 1080, we need 540 threads
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//// for each row
	//int idy = blockIdx.y * blockDim.y + threadIdx.y;

	//int idxEven = idx * 2 + idy * w;
	//int idxOdd = idx * 2 + 1 + idy * w;

	//int offset = 1;

	//temp[2 * tdx] = input[idxEven];
	//temp[2 * tdx + 1] = input[idxOdd];

	//for (int nSum = B_SIZE / 2; nSum > 0; nSum /= 2)
	//{ 
	//	__syncthreads();
	//	if (tdx < nSum)
	//	{
	//		int a = offset * (2 * tdx + 1) - 1;
	//		int b = offset * (2 * tdx + 2) - 1;
	//		temp[b] += temp[a];
	//	}
	//	offset *= 2;
	//}

	//__syncthreads();

	////Write output (size h)
	//output[2 * tdx] = temp[2 * tdx];
	//output[2 * tdx + 1] = temp[2 * tdx + 1];
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
		return (float)((im[index + 1] - im[index - 1]) / 2);
	}
	else if (col_index + 1 == width)
	{
		return (float)((im[index] - im[index - 1]) / 2);
	}
	else if (col_index - 1 == 0)
	{
		return (float)((im[index + 1] - im[index]) / 2);
	}
}

__host__ float x_derivativeCPU(unsigned char* im, int col_index, int index, int width) {
	if ((col_index + 1) < width && (col_index - 1) >= 0)
	{
		return ((float)(im[index + 1] - im[index - 1]) / 2);
	}
	else if (col_index + 1 >= width)
	{
		return ((float)(im[index] - im[index - 1]) / 2);
	}
	else
	{
		return ((float)(im[index + 1] - im[index]) / 2);
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

__global__ void x_derivativeOnGPU(unsigned char* in, float* out, int w, int h) {
	int id = blockIdx.x*blockDim.y + threadIdx.y;
	int idx = id % w;
	if (id >= w * h) return;
	if ((idx - 1) >= 0 && idx + 1 < w) { 
		out[id] = ((int) in[id + 1] - (int) in[id - 1])*1.0f / 2; 
	}
	else if (idx + 1 == w) {
		out[id] = ((int) in[id] - (int)in[id-1])*1.0f / 2;
	}
	else if (idx - 1 == 0) { 
		out[id] = ((int)in[id+1] - (int) in[id])*1.0f / 2;
	}
}
/**
__global__ void x_derivativeOnGPU(unsigned char* in, unsigned char* out, const int w, const int h) {
int tdx = threadIdx.x;
int id = blockIdx.x*blockDim.y + threadIdx.y;
int idx = id % w;
int idy = id / w;
__shared__ float s_f[TILE_WIDTH][3];
for (int i = 0; i < TILE_WIDTH; i++) {
if ((idx - 1) >= 0 && idx + 1 < w) {
s_f[i][0] = in[idx - 1];
s_f[i][1] = in[idx];
s_f[i][2] = in[idx + 1];
}
else if (idx + 1 == w) {
s_f[i][0] = in[idx - 1];
s_f[i][1] = in[idx];
s_f[i][2] = in[idx];
}
else if (idx - 1 == 0) {
s_f[i][0] = in[idx];
s_f[i][1] = in[idx];
s_f[i][2] = in[idx + 1];
}
__syncthreads();
s_f[i][1] = (s_f[i][2] - s_f[i][0]) / 2;
out[id] = s_f[i][1];

}

}
**/