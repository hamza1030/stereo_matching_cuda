#pragma once

#include "SystemIncludes.h"



void compute_cost(unsigned char* i1, unsigned char* i2, float* cost, int w1, int w2, int h1, int h2);
__global__ void costVolumOnGPU(unsigned char* i1, unsigned char* i2, float* cost,int w1, int w2,  int h1,  int h2, int size_d);
__device__ float x_derivative(unsigned char* im, int col_index, int index, int width);
__device__ int difference_term(unsigned char pixel_i, unsigned char pixel_j);
__device__ int difference_term_2(float pixel_i, float pixel_j);
__host__ int iDivUp(int a, int b);
__device__ int id_cost(int i, int j, int width, int height, int k);
__device__ int id_im(int i, int j, int width);