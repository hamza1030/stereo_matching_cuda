#include "SystemIncludes.h"



__global__ void occlusionOnGpu(unsigned char* image, unsigned char* mean, int width, int height);
void detect_occlusion(unsigned char* il, unsigned char* ir, float* cost, int w1, int w2, int h1, int h2, bool host_gpu_compare)