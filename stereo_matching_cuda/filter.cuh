#include "SystemIncludes.h"
#include "helpers.cuh"
#define B_H (TILE_HEIGHT +2*RADIUS)
#define B_W (TILE_WIDTH +2*RADIUS)

void boxFilterOnCPU(unsigned char* image, unsigned char* mean, int width, int height);

__global__ void boxFilterOnGPU(unsigned char* image, unsigned char* mean, int width, int height);
__global__ void multIm(unsigned char* im1, unsigned char* im2, float* val, int width, int height);
__global__ void boxFilterfloatOnGpu(float* image, float* mean, int width, int height);
__global__ void sousIm(float* im1, float* im2, float* val, int width, int height);
void filter(unsigned char* image, int width, int height, unsigned char* mean, float* var, bool cuda);
