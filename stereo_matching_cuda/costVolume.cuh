#pragma once

#include "SystemIncludes.h"



void compute_cost(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2);
bool check_errors_cost(float* resCPU, float* resGPU, int len);
void costVolumeOnCPU(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2, int size_d);
__host__ float x_derivativeCPU(unsigned char* im, int col_index, int index, int width);
__host__ int iDivUp(int a, int b);
__global__ void costVolumOnGPU2(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2, int size_d);
__device__ float x_derivative(unsigned char* im, int col_index, int index, int width);
__device__ int difference_term(unsigned char pixel_i, unsigned char pixel_j);
__device__ float difference_term_2(float pixel_i, float pixel_j);
__device__ int id_cost(int i, int j, int width, int height, int k);
__device__ int id_im(int i, int j, int width);
__device__ int getGlobalIdx_1D_2D();