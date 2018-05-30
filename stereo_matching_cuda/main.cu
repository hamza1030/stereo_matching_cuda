#include "rgb_to_grayscale.cuh"
#include "stb_image.h"
#include "stb_image_write.h"
#include "filter.cuh"
#include "costVolume.cuh"
#include "systemIncludes.h"
#include "guidedFilter.cuh"
#include "occlusion.cuh"

// int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
// DOCUMENTATION
//
// Limitations:
//    - no 12-bit-per-channel JPEG
//    - no JPEGs with arithmetic coding
//    - GIF always returns *comp=4
//
// Basic usage (see HDR discussion below for HDR usage):
//    int x,y,n;
//    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
//    // ... process data if not NULL ...
//    // ... x = width, y = height, n = # 8-bit components per pixel ...
//    // ... replace '0' with '1'..'4' to force that many components per pixel
//    // ... but 'n' will always be the number that it would have been if you said 0
//    stbi_image_free(data)
//
// Standard parameters:
//    int *x                 -- outputs image width in pixels
//    int *y                 -- outputs image height in pixels
//    int *channels_in_file  -- outputs # of image components in image file
//    int desired_channels   -- if non-zero, # of image components requested in result
//
// The return value from an image loader is an 'unsigned char *' which points
// to the pixel data, or NULL on an allocation failure or if the image is
// corrupt or invalid. The pixel data consists of *y scanlines of *x pixels,
// with each pixel consisting of N interleaved 8-bit components; the first
// pixel pointed to is top-left-most in the image. There is no padding between
// image scanlines or between pixels, regardless of format. The number of
// components N is 'desired_channels' if desired_channels is non-zero, or
// *channels_in_file otherwise. If desired_channels is non-zero,
// *channels_in_file has the number of components that _would_ have been
// output otherwise. E.g. if you set desired_channels to 4, you will always
// get RGBA output, but you can check *channels_in_file to see if it's trivially
// opaque because e.g. there were only 3 channels in the source image.
//
// An output image with N components has the following components interleaved
// in this order in each pixel:
//
//     N=#comp     components
//       1           grey
//       2           grey, alpha
//       3           red, green, blue
//       4           red, green, blue, alpha
//
// If image loading fails for any reason, the return value will be NULL,
// and *x, *y, *channels_in_file will be unchanged. The function
// stbi_failure_reason() can be queried for an extremely brief, end-user
// unfriendly explanation of why the load failed. Define STBI_NO_FAILURE_STRINGS
// to avoid compiling these strings at all, and STBI_FAILURE_USERMSG to get slightly
// more user-friendly ones.
//
// Paletted PNG, BMP, GIF, and PIC images are automatically depalettized.

int main(int argc, char **argv)
{
	bool host_compare = true;
	int wRadius = 3;

	//// Image loading
	int width;
	int height;
	int channels_in_file;

	unsigned char *data = stbi_load("im2.png", &width, &height, &channels_in_file, 0);

	//// Split the image in several channels

	// Make an array for each channel
	unsigned char **splitted_data = new unsigned char *[channels_in_file];
	for (size_t channel = 0; channel < channels_in_file; ++channel)
	{
		splitted_data[channel] = new unsigned char[width*height];
	}

	// Fill the arrays
	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			for (size_t channel = 0; channel < channels_in_file; ++channel)
			{
				splitted_data[channel][row * width + col] = data[channel + (row * (width * channels_in_file) + col * channels_in_file)];
			}
		}
	}

	printf("Starting...\n");

	// set up devices
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	//end setup device

	//Warmup
	int n = width * height;
	cout << "Warmup ..." << endl;
	unsigned char* grayscale = rgb_to_grayscale(data, n, channels_in_file, !host_compare);
	stbi_write_png("./data/warmup.png", width, height, 1, grayscale, 0);
	//end warmup

	/**
	///////////////////////////////////////////////////////////////////////////
	// Write each channel in separated files
	for (size_t channel = 0; channel < channels_in_file; ++channel)
	{
		std::string filename("image_" + std::to_string(channel) + ".png");
		stbi_write_png(filename.c_str(), width, height, 1, splitted_data[channel], 0);
	}

	//// Reconstruct the original image from splitted images
	unsigned char *copy_data = new unsigned char[channels_in_file * width * height];
	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			for (size_t channel = 0; channel < channels_in_file; ++channel)
			{
				copy_data[channel + (row * (width * channels_in_file) + col * channels_in_file)] = splitted_data[channel][(row * width + col)];
			}
		}
	}

	stbi_write_png("image_copy.png", width, height, channels_in_file, copy_data, 0);

	//// Free the memory
	for (int channel = 0; channel < channels_in_file; ++channel)
	{
		delete[] splitted_data[channel];
	}
	delete[] splitted_data;

	delete copy_data;
	////////////////////////////////////////////////////////////////////////////////////////////////
	**/

	//Begin
	// Image loading
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
	memset(costl, 0, sizeof(float)*totalSize1);
	memset(costr, 0, sizeof(float)*totalSize2);
	cout << "Cost Volume ..." << endl;

	const int dminl = D_MIN;
	compute_cost(I_l, I_r, costl, w1, w2, h1, h2, dminl, host_compare);
	const int dminr = -D_MAX;
	compute_cost(I_r, I_l, costr, w2, w1, h2, h1, dminr, host_compare);
	//end cost volume

	//guided Filter
	unsigned char* mean1 = (unsigned char*)malloc(n1);
	unsigned char* mean2 = (unsigned char*)malloc(n2);
	float* filtered_costl = (float*)malloc(sizeof(float) * totalSize1);
	float* filtered_costr = (float*)malloc(sizeof(float) * totalSize2);
	memset(mean1, 0, sizeof(unsigned char)*n1);
	memset(mean2, 0, sizeof(unsigned char)*n2);
	memset(filtered_costl, 0, sizeof(float)*totalSize1);
	memset(filtered_costr, 0, sizeof(float)*totalSize2);
	cout << "guided filter ..." << endl;
	compute_guided_filter(I_l, costl, filtered_costl, mean1, (const int)w1, (const int)h1, (const int)size_d, host_compare);
	compute_guided_filter(I_r, costr, filtered_costr, mean2, (const int)w2, (const int)h2, (const int)size_d, host_compare);

	//unsigned char* mean = (unsigned char*)malloc(height*width); //osef
	//memset(mean, 0, sizeof(unsigned char)*totalSize1);
	//compute_guided_filter(grayscale, cost, filtered, mean, (const int)width, (const int)height, (const int)size_d, host_compare);
	//stbi_write_png("./data/uhd_mean.png", width, height, 1, mean, 0);
	//free(mean);


	//end guided Filter

	float* best_costl = (float*)malloc(n1 * sizeof(float));
	float* best_costr = (float*)malloc(n2 * sizeof(float));
	memset(best_costl, 9999999.0f, n1 * sizeof(float));
	memset(best_costr, 9999999.0f, n2 * sizeof(float));

	float* dmapl = (float*)malloc(n1 * sizeof(float));
	float* dmapr = (float*)malloc(n2 * sizeof(float));
	unsigned char* dmaplChar = (unsigned char*)malloc(n1);
	unsigned char* dmaprChar = (unsigned char*)malloc(n2);
	memset(dmapl, 0, n1 * sizeof(float));
	memset(dmapr, 0, n2 * sizeof(float));
	memset(dmaplChar, 0, n1);
	memset(dmaprChar, 0, n2);
	disparity_selection(filtered_costl, best_costl, dmapl, (const int)w1, (const int)h1, dminl, host_compare);
	disparity_selection(filtered_costr, best_costr, dmapr, (const int)w2, (const int)h2, dminr, host_compare);
	//for (int i = 0; i < n1; i++) { cout << best_costl[i] << endl; }

	//const int dOcclusion = 2 * size_d;
	const int dOcclusion = (dminl - 1);
	detect_occlusion(dmapl, dmapr, dOcclusion, dmaplChar, dmaprChar, w1, h1);
	int vMin = D_MIN;
	fill_occlusion(dmapl, w1, h1, vMin);

	//write images
	cout << "writing images ..." << endl;
	stbi_write_png("./data/image_left.png", w1, h1, 1, I_l, 0);
	stbi_write_png("./data/image_right.png", w2, h2, 1, I_r, 0);
	stbi_write_png("./data/image_mean_left.png", w1, h1, 1, mean1, 0);
	stbi_write_png("./data/image_mean_right.png", w2, h2, 1, mean2, 0);
	stbi_write_png("./data/disparity_map_left.png", w1, h1, 1, dmaplChar, 0);
	stbi_write_png("./data/disparity_map_right.png", w2, h2, 1, dmaprChar, 0);
	//end writing images

	//free the memory
	cout << "Free the memory ..." << endl;
	free(grayscale);
	stbi_image_free(data);
	free(I_l);
	free(I_r);
	stbi_image_free(data1);
	stbi_image_free(data2);
	free(mean1);
	free(mean2);
	free(costl);
	free(costr);
	free(filtered_costl);
	free(filtered_costr);
	free(dmapl);
	free(dmapr);
	free(best_costr);
	free(best_costl);
	free(dmaprChar);
	free(dmaplChar);

	return 0;
}