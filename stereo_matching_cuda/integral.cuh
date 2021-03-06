#include "SystemIncludes.h"

void integral(float* image, float* integral, int width, int height);
__global__ void transpose(float* in, float* out, const int w, const int h);
__global__ void rowSum(float * in, float * out, const int w, const int h);
__global__ void colSum(float * in, float * out, const int w, const int h);
void integralOnCPU(float * in, float * out, const int w, const int h);
__global__ void rowSum_sm(float* in, float* out, const int w, const int h);