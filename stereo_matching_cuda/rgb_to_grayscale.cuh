#pragma once

#include "SystemIncludes.h"

void sumArraysOnHost(unsigned char* image, unsigned char* gray, const int N, int channels);
__global__ void sumArraysOnGPU(unsigned char* image, unsigned char* gray, const int N, int channels);
unsigned char* rgb_to_grayscale(unsigned char* h_rgb, const int n, int channels,bool cuda);
