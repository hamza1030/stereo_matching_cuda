#include "rgb_to_grayscale.cuh"
#include "filter.cuh"
#include "costVolume.cuh"
#include "systemIncludes.h"
#include "guidedFilter.cuh"
#include "occlusion.cuh"
#include "helpers.cuh"
#include "stb_image.h"
#include "stb_image_write.h"



void write_mat(float* mat, const char* filename, int w, int h, int start) {
	unsigned char* matchar = (unsigned char*)malloc(w*h);
	memset(matchar, 0, w*h);
	float max = -150000000.0f;
	float min = 150000000.0f;
	for (int i = start; i < start + w * h; i++) {
		if (mat[i] > max) {
			max = mat[i];
		}
		else if (mat[i] <= min) {
			min = mat[i];

		}
	}
	for (int i = 0; i < w*h; i++) {
		int c = (mat[i + start] - min) * 255.0f / (max - min);
		//cout << c << endl;
		matchar[i] = (unsigned char)c;
	}
	stbi_write_png(filename, w, h, 1, matchar, 0);
	free(matchar);

}

int main(int argc, char **argv)
{

		bool host_compare = false;
		printf("Starting...\n");

		// set up devices
		int dev = 0;
		cudaDeviceProp deviceProp;
		CHECK(cudaGetDeviceProperties(&deviceProp, dev));
		printf("Using Device %d: %s\n", dev, deviceProp.name);
		CHECK(cudaSetDevice(dev));
		//end setup device
		//Begin
		// Image loading
		std::clock_t start;
		double duration;
		start = std::clock();
		int w1, h1, ch1;
		int w2, h2, ch2;
		unsigned char *data1 = stbi_load("./data/tsukuba0.png", &w1, &h1, &ch1, 0);
		unsigned char *data2 = stbi_load("./data/tsukuba1.png", &w2, &h2, &ch2, 0);
		int n1 = w1 * h1;
		int n2 = w2 * h2;

		cout << "Resolution : " << w1 << "x" << h1 << endl;
		//rgb to grayscale
		cout << "RGB to grayscale ..." << endl;
		unsigned char* I_l = rgb_to_grayscale(data1, n1, ch1, host_compare);
		unsigned char* I_r = rgb_to_grayscale(data2, n2, ch2, host_compare);
		//end rgb to grayscale

		//Cost volume
		int size_d = D_MAX - D_MIN + 1;
		int totalSize1 = n1 * size_d;
		int totalSize2 = n2 * size_d;
		float* costl = (float*)malloc(sizeof(float)*totalSize1);
		float* costr = (float*)malloc(sizeof(float)*totalSize2);
		memset(costl, 0.0f, sizeof(float)*totalSize1);
		memset(costr, 0.0f, sizeof(float)*totalSize2);
		cout << "Cost Volume ..." << endl;

		const int dminl = D_MIN;
		compute_cost(I_l, I_r, costl, w1, w2, h1, h2, dminl, host_compare);
		const int dminr = -D_MAX;
		compute_cost(I_r, I_l, costr, w2, w1, h2, h1, dminr, host_compare);

		//end cost volume

		//guided Filter
		unsigned char* mean1 = (unsigned char*)malloc(n1);
		unsigned char* mean2 = (unsigned char*)malloc(n2);
		unsigned char* h_mean1 = (unsigned char*)malloc(n1);
		unsigned char* h_mean2 = (unsigned char*)malloc(n2);
		float* filtered_costl = (float*)malloc(sizeof(float) * totalSize1);
		float* filtered_costr = (float*)malloc(sizeof(float) * totalSize2);
		memset(mean1, 0, sizeof(unsigned char)*n1);
		memset(mean2, 0, sizeof(unsigned char)*n2);
		memset(h_mean1, 0, sizeof(unsigned char)*n1);
		memset(h_mean2, 0, sizeof(unsigned char)*n2);
		memset(filtered_costl, 0, sizeof(float)*totalSize1);
		memset(filtered_costr, 0, sizeof(float)*totalSize2);


		//unsigned char* mean = (unsigned char*)malloc(height*width); //osef
		//memset(mean, 0, sizeof(unsigned char)*totalSize1);
		//compute_guided_filter(grayscale, cost, filtered, mean, (const int)width, (const int)height, (const int)size_d, host_compare);
		//stbi_write_png("./data/uhd_mean.png", width, height, 1, mean, 0);
		//free(mean);


		float* best_costl = (float*)malloc(n1 * sizeof(float));
		float* best_costr = (float*)malloc(n2 * sizeof(float));
		float* h_best_costl = (float*)malloc(n1 * sizeof(float));
		float* h_best_costr = (float*)malloc(n2 * sizeof(float));
		memset(best_costl, 9999999.0f, n1 * sizeof(float));
		memset(best_costr, 9999999.0f, n2 * sizeof(float));
		memset(h_best_costl, 9999999.0f, n1 * sizeof(float));
		memset(h_best_costr, 9999999.0f, n2 * sizeof(float));

		float* dmapl = (float*)malloc(n1 * sizeof(float));
		float* dmapr = (float*)malloc(n2 * sizeof(float));
		memset(dmapl, 0, n1 * sizeof(float));
		memset(dmapr, 0, n2 * sizeof(float));

		float* h_dmapl = (float*)malloc(n1 * sizeof(float));
		float* h_dmapr = (float*)malloc(n2 * sizeof(float));
		memset(h_dmapl, 0, n1 * sizeof(float));
		memset(h_dmapr, 0, n2 * sizeof(float));

		unsigned char* dmaplChar = (unsigned char*)malloc(n1);
		unsigned char* dmaprChar = (unsigned char*)malloc(n2);
		memset(dmaplChar, 0, n1);
		memset(dmaprChar, 0, n2);

		cout << "guided filter ..." << endl;
		compute_guided_filter(I_l, costl, best_costl, dmapl, mean1, (const int)w1, (const int)h1, (const int)size_d, dminl, host_compare);
		compute_guided_filter(I_r, costr, best_costr, dmapr, mean2, (const int)w2, (const int)h2, (const int)size_d, dminr, host_compare);

		if (host_compare) {
			guided_filter_onCpu(I_l, costl, h_best_costl, h_dmapl, h_mean1, w1, h1, size_d, dminl);
			guided_filter_onCpu(I_r, costr, h_best_costr, h_dmapr, h_mean2, w2, h2, size_d, dminr);
		}
		float* occlusion = (float*)malloc(n1*sizeof(float));
		memcpy(occlusion, dmapl, n1 * sizeof(float));
		float* occlusion_filled = (float*)malloc(n1 * sizeof(float));
		


		cout << "guided filter ok" << endl;
	
		//detect occlusion
		const int dOcclusion = (dminl-100);
		detect_occlusion(occlusion, dmapr, dOcclusion, dmaplChar, dmaprChar, w1, h1);

		//simple filling
		memcpy(occlusion_filled, occlusion, n1 * sizeof(float));
		int vMin = D_MIN;
		fill_occlusion(occlusion_filled, w1, h1, vMin);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;



		//write images
		cout << "writing images ..." << endl;
		stbi_write_png("./data/image_left.png", w1, h1, 1, I_l, 0);
		stbi_write_png("./data/image_right.png", w2, h2, 1, I_r, 0);
		stbi_write_png("./data/image_mean_left.png", w1, h1, 1, mean1, 0);
		stbi_write_png("./data/image_mean_right.png", w2, h2, 1, mean2, 0);
		const char* bcostlname = "./data/best_costl.png";
		write_mat(best_costl, bcostlname, w1, h1, 0);
		const char* bcostrname = "./data/best_costr.png";
		write_mat(best_costr, bcostrname, w2, h2, 0);
		const char* costnamel = "./data/cost_lminus15.png";
		write_mat(costl, costnamel, w1, h1, 0);
		const char* costnamer = "./data/cost_rminus15.png";
		write_mat(costr, costnamer, w2, h2, 0);
		const char* occlul = "./data/occlu_mapl.png";
		write_mat(occlusion, occlul, w1, h1, 0);
		const char* dmaplname = "./data/disparity_mapl.png";
		write_mat(dmapl, dmaplname, w1, h1, 0);
		const char* dmaprname = "./data/disparity_mapr.png";
		write_mat(dmapr, dmaprname, w2, h2, 0);
		const char* occlul_filled = "./data/occlu_mapl_filled.png";
		write_mat(occlusion_filled, occlul_filled, w1, h1, 0);
		//end writing images

		std::cout << "duration: " << duration << std::endl;
		//free the memory
		cout << "Free the memory ..." << endl;
		free(occlusion);
		free(occlusion_filled);
		free(I_l);
		free(I_r);
		stbi_image_free(data1);
		stbi_image_free(data2);
		free(mean1);
		free(mean2);
		free(h_mean1);
		free(h_mean2);
		free(costl);
		free(costr);
		free(filtered_costl);
		free(filtered_costr);
		free(dmapl);
		free(dmapr);
		free(h_dmapl);
		free(h_dmapr);
		free(best_costr);
		free(best_costl);
		free(dmaprChar);
		free(dmaplChar);
		free(h_best_costl);
		free(h_best_costr);

	
	return 0;
}