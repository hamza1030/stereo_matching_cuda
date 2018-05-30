#include "helpers.cuh"

bool check_errors(float* resCPU, float* resGPU, int len) {
	cout << "KEKE" << endl;
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

bool check_errors(unsigned char* resCPU, unsigned char* resGPU, int len) {
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