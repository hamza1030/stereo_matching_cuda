#include "SystemIncludes.h"


void compute_guided_filter(unsigned char* i1, unsigned char* i2, float* cost, unsigned char* mean1, unsigned char* mean2, int w1, int h1, int w2, int h2, int size_d, bool host_gpu_compare);


__host__ void chToFlOnCPU(unsigned char* image, float* result, int len);
__host__ void flToChOnCPU(float* image, unsigned char* result, int len);
__host__ void pixelMultOnCPU(float* image1, float* image2, float* result, int len);
__host__ void pixelSousOnCPU(float* image1, float* image2, float* result, int len);
__host__ void pixelAddOnCPU(float* image1, float* image2, float* result, int len);
__host__ void pixelDivOnCPU(float* image1, float* image2, float* result, int len);


__global__ void chToFlOnGPU(unsigned char* image, float* result, int len);
__global__ void flToChOnGPU(float* image, unsigned char* result, int len);
__global__ void pixelMultOnGPU(float* image1, float* image2, float* result, int len);
__global__ void pixelSousOnGPU(float* image1, float* image2, float* result, int len);
__global__ void pixelAddOnGPU(float* image1, float* image2, float* result, int len);
__global__ void pixelDivOnGPU(float* image1, float* image2, float* result, int len);
__global__ void compute_mean_x(float *image, float *mean, int w, int h, int radius);
__global__ void compute_mean_y(float *image, float *mean, int w, int h, int radius);

__device__ void mean_x(float *id, float *od, int w, int h, int r);
__device__ void mean_y(float *id, float *od, int w, int h, int r);