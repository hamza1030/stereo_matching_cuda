#include "helpers.cuh"

bool check_errors(float* resCPU, float* resGPU, int len) {
	bool res = true;
	for (int i = 0; i < len; i++) {
		//cout << i << " ResultGPU = " << resGPU[i] << " and ResultCPU= " << resCPU[i] << endl;
		if (resCPU[i] != resGPU[i]) {
			res = false;
			cout << "error at element: " << i << " ResultGPU = " << resGPU[i] << " and ResultCPU= " << resCPU[i] << endl;
		}
	}
	return res;
}

bool check_errors(float* resCPU, unsigned char* resGPU, int len) {
	bool res = true;
	for (int i = 0; i < len; i++) {
		if (resCPU[i] != resGPU[i]) {
			res = false;
			cout << "error at element: " << i << " ResultGPU = " << resGPU[i] << " and ResultCPU= " << resCPU[i] << endl;
		}
	}
	return res;
}