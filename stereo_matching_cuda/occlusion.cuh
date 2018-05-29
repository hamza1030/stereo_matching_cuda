#include "SystemIncludes.h"


// dOcclusion = (dMin - 1)
// dLR = 0

void detect_occlusion(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h);
__global__ void detect_occlusionOnGPU(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h);

void detect_occlusionOnCPU(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h);

void fill_occlusion(float* disparity, const int w, const int h, const float vMin);
__global__ void fill_occlusionOnGPU1(float* disparity, const int w, const int h, const float vMin);
__global__ void fill_occlusionOnGPU2(float* disparity, const int w, const int h, const float vMin);


void fill_occlusionOnCPU(float* disparity, const int w, const int h, const float vMin);
