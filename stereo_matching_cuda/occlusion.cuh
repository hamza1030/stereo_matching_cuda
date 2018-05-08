#include "SystemIncludes.h"



void detect_occlusion(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h);
__global__ void detect_occlusionOnGPU(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h);

void detect_occlusionOnCPU(float* disparityLeft, float* disparityRight, const float dOcclusion, const int dLR, const int w, const int h);

