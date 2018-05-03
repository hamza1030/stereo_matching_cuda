#pragma once

#include "SystemIncludes.h"

void sumArraysOnHost(unsigned char* image, unsigned char* gray, const int N, int channels);
__global__ void sumArraysOnGPU(unsigned char* image, unsigned char* gray, const int N, int channels);
unsigned char* rgb_to_grayscale(unsigned char* h_rgb, const int n, int channels, bool host_gpu_compare);
__host__ bool check_errors_grayscale(unsigned char * host, unsigned char * gpu, int len);