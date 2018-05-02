#include "SystemIncludes.h"



__global__ void boxFilterOnGpu(unsigned char* image, unsigned char* mean, int width, int height);
__global__ void multIm(unsigned char* im1, unsigned char* im2, float* val, int width, int height);
__global__ void boxFilterfloatOnGpu(float* image, float* mean, int width, int height);
__global__ void sousIm(float* im1, float* im2, float* val, int width, int height);
void filter(unsigned char* image, int width, int height, unsigned char* mean, float* var, bool cuda);
