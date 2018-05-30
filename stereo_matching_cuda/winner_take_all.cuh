
/*

// GPU functions

__device__ void mean_x(float *id, float *od, int w, int h, int r)
{
float scale = 1.0f / (float)((r << 1) + 1);

float t;
// do left edge
t = id[0] * r;

for (int x = 0; x < (r + 1); x++)
{
t += id[x];
}

od[0] = t * scale;

for (int x = 1; x < (r + 1); x++)
{
t += id[x + r];
t -= id[0];
od[x] = t * scale;
}

// main loop
for (int x = (r + 1); x < w - r; x++)
{
t += id[x + r];
t -= id[x - r - 1];
od[x] = t * scale;
}

// do right edge
for (int x = w - r; x < w; x++)
{
t += id[w - 1];
t -= id[x - r - 1];
od[x] = t * scale;
}
}

__device__ void mean_y(float *id, float *od, int w, int h, int r)
{
float scale = 1.0f / (float)((r << 1) + 1);

float t;
// do left edge
t = id[0] * r;

for (int y = 0; y < (r + 1); y++)
{
t += id[y * w];
}

od[0] = t * scale;

for (int y = 1; y < (r + 1); y++)
{
t += id[(y + r) * w];
t -= id[0];
od[y * w] = t * scale;
}

// main loop
for (int y = (r + 1); y < (h - r); y++)
{
t += id[(y + r) * w];
t -= id[((y - r) * w) - w];
od[y * w] = t * scale;
}

// do right edge
for (int y = h - r; y < h; y++)
{
t += id[(h - 1) * w];
t -= id[((y - r) * w) - w];
od[y * w] = t * scale;
}
}

__global__ void compute_mean_x(float *image, float *mean, int w, int h, int radius)
{
unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
mean_x(&image[y * w], &mean[y * w], w, h, radius);
}

__global__ void compute_mean_y(float *image, float *mean, int w, int h, int radius)
{
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
mean_y(&image[x], &mean[x], w, h, radius);
}

*/