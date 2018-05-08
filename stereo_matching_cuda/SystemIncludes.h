#pragma once

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define CLOCKS_PER_SEC 1000
#define R_W 0.299
#define G_W 0.587
#define B_W 0.0721
#define ALPHA 0.9
#define D_MAX 3
#define D_MIN 0
#define TH_grad 2
#define TH_color 7

#define SIZE_1D 100
#define B_SIZE 64

#define TILE_HEIGHT 12
#define TILE_WIDTH 12
#define RADIUS 9



#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <string>
#include <time.h>

using namespace std;

#define CHECK(call) \
do { \
	if (cudaSuccess != call) { \
		fprintf(stderr, ("CUDA ERROR! file: %s[%i] -> %s\n"), __FILE__, __LINE__, cudaGetErrorString(call)); \
		exit(0); \
	} \
} while (0)