#include "costVolume.cuh"
__host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
void compute_cost(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2) {
	int size_d = D_MAX - D_MIN + 1;
	int size_cost = h1 * w1*size_d;
	memset(cost, 0, sizeof(float)*(size_cost));

	unsigned char* d_i1;
	unsigned char* d_i2;

	float* d_cost;

	CHECK(cudaMalloc((unsigned char**)&d_i1, w1 * h1));
	CHECK(cudaMalloc((unsigned char**)&d_i2, w2 * h2));
	CHECK(cudaMalloc((void**)&d_cost, size_cost * sizeof(float)));
	CHECK(cudaMemcpy(d_i1, i1, w1 * h1, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_i2, i2, w2 * h2, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_cost, cost, sizeof(float)*size_cost, cudaMemcpyHostToDevice));
	dim3 blockDim(12, 12, size_d);
	dim3 gridDim;
	gridDim.x = iDivUp( w1*h1*size_d,blockDim.x);
	gridDim.y = iDivUp( h1*w1*size_d, blockDim.y);
	gridDim.z = 1;

	costVolumOnGPU << <gridDim, blockDim >> > (d_i1, d_i2, d_cost, w1, w2, h1, h2, size_d);

	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(cost, d_cost, size_cost * sizeof(float), cudaMemcpyDeviceToHost));

	// free device global memory
	CHECK(cudaFree(d_cost));
	CHECK(cudaFree(d_i1));
	CHECK(cudaFree(d_i2));
}
__global__ void costVolumOnGPU(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2, int size_d) {
	/**
		i1: left image
		w1: width of left image
		h1: height of left image
		i2: right image
		w2: width of left image
		h2: height of left image
		cost: cost volume
		d: disparity
	**/
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int z = blockDim.z*blockIdx.z + threadIdx.z;


	//int offset = D_MIN + z; //d = d_min + z <=d_max
	//int x_2 = x + offset;//ix_right_image = (ix_left_image + d)
	int index_1 = id_im(x,y,w1);// (ix,iy) => 1D: i = iy*width + ix
	int index_2 = id_im(x + D_MIN +z,y,w2);// i_right = i_y*width + ix_right_image
	int index_cost = id_cost(x,y,w1,h1,z); //volume (3D): for the zth number of the range [d_min, d_max] => i_cost = z*width*height + i
	if (z < size_d && x < w1 && y < h1) {
		// z in [0, len(d)[, x in [0, width[, y in [0, height[
		if (x + z + D_MIN < w1 && x + z + D_MIN >= 0) {

			cost[index_cost] = (1 - ALPHA) * difference_term(i1[index_1], i2[index_2]) + ALPHA * difference_term_2(x_derivative(i1, x, index_1, w1), x_derivative(i2, x + D_MIN +z, index_2, w2));
			printf("(%i,%i,%i) -> (%i,%i,%i) -> %f\n", x, y, z, index_1,index_2,index_cost, cost[index_cost]);
		}
		else {
			printf("out of bounds (%i,%i,%i) -> (%i,%i,%i)\n -> %f\n", x, y, z, index_1, index_2, index_cost, cost[index_cost]);
			cost[index_cost] =(float) ((1 - ALPHA) * TH_color + ALPHA * TH_grad);
			
		}
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

__device__ int difference_term(unsigned char pixel_i, unsigned char pixel_j) {
	return min(abs((int)(pixel_i - pixel_j)), TH_color);
}
__device__ int difference_term_2(float pixel_i, float pixel_j) {
	return min(abs(int(pixel_i - pixel_j)), TH_color);
}