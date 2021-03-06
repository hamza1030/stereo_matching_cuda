#include "SystemIncludes.h"
#include "helpers.cuh"


// dOcclusion = (dMin - 1)
// dLR = 0

void detect_occlusion(float* disparityLeft, float* disparityRight, const int dOcclusion,unsigned char* dmapl ,unsigned char* dmapr ,const int w, const int h);
__global__ void detect_occlusionOnGPU(float* disparityLeft, float* disparityRight, const int dOcclusion, const int w, const int h);

__global__ void flToCh2OnGPU(float* image, unsigned char* result, int min, int max, int len, const int dOcclusion);
void detect_occlusionOnCPU(float* disparityLeft, float* disparityRight, const int dOcclusion, const int w, const int h);

void fill_occlusion(float* disparity, const int w, const int h, const float vMin);
__global__ void fill_occlusionOnGPU1(float* disparity, const int w, const int h, const float vMin);
__global__ void fill_occlusionOnGPU2(float* disparity, const int w, const int h, const float vMin);


void fill_occlusionOnCPU(float* disparity, const int w, const int h, const float vMin);
