#pragma once

#include "SystemIncludes.h"

bool check_errors(float* resCPU, float* resGPU, int len);
bool check_errors(unsigned char* resCPU, unsigned char* resGPU, int len);