#pragma once

#include "SystemIncludes.h"
#include "integral.cuh"
#include "helpers.cuh"

void compute_guided_filter(unsigned char* i, float* cost, float* filter_cost, float* disp_map, unsigned char* mean, const int w, const int h, const int size_d, int dmin, bool host_gpu_compare);
__global__ void dispSelectOnGPU(float* q, float* filter_cost, float* dmap, const int n, int label);




__host__ void chToFlOnCPU(unsigned char* image, float* result, int len);
__host__ void flToChOnCPU(float* image, unsigned char* result, int len);
__host__ void pixelMultOnCPU(float* image1, float* image2, float* result, int len);
__host__ void pixelSousOnCPU(float* image1, float* image2, float* result, int len);
__host__ void pixelAddOnCPU(float* image1, float* image2, float* result, int len);
__host__ void pixelDivOnCPU(float* image1, float* image2, float* result, int len);

__global__ void computeBoxFilter(float* image, float* integral, float* mean, const int w, const int h);
__global__ void chToFlOnGPU(unsigned char* image, float* result, int len);
__global__ void flToChOnGPU(float* image, unsigned char* result, int len);
__global__ void pixelMultOnGPU(float* image1, float* image2, float* result, int len);
__global__ void pixelSousOnGPU(float* image1, float* image2, float* result, int len);
__global__ void pixelAddOnGPU(float* image1, float* image2, float* result, int len);
__global__ void pixelDivOnGPU(float* image1, float* image2, float* result, int len);
__device__ float computeMean(float* I, float* S, int idx, int idy, const int w, const int h);

__global__ void copyFromLittleToBigOnGPU(float* image1, float* result, int start, int len);
__global__ void copyFromBigToLittleOnGPU(float* image1, float* result, int start, int len);
__global__ void compute_ak(float* mean, float* var, float* convol, float* pk, float* a, int len);
__global__ void compute_bk(float* mean, float* a, float* pk, float* b, int len);
__global__ void compute_ak_and_bk(float* mean, float* var, float* convol, float* pk, float* a, float* b, int len);
__global__ void compute_q(float* im, float* a, float* b, float* q, int len);
/**
__global__ void compute_mean_x(float *image, float *mean, int w, int h, int radius);
__global__ void compute_mean_y(float *image, float *mean, int w, int h, int radius);

__device__ void mean_x(float *id, float *od, int w, int h, int r);
__device__ void mean_y(float *id, float *od, int w, int h, int r);
**/