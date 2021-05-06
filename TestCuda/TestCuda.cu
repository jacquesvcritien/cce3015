#include <cassert>

__global__ void MatAdd(int n, float *c, const float *a, const float *b, int pitch)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ij = i * pitch + j;
	if(i < n && j < n)
		c[ij] = a[ij] + b[ij];
}

int main()
{
	const int n = 1024;
	float c[n][n], a[n][n], b[n][n];

	for(int i=0; i < n; i++){
		for(int j=0; j < n; j++){
			a[i][j] = 1;
			b[i][j] = 2;
			c[i][j] = 3;
		}
	}

	printf("Filled arrays");

	const int rowsize = n * sizeof(float);
	float *dc, *da, *db;

	size_t pitch;
	cudaMallocPitch((void**)&da, &pitch, rowsize, n);
	cudaMallocPitch((void**)&db, &pitch, rowsize, n);
	cudaMallocPitch((void**)&dc, &pitch, rowsize, n);


	printf("Malloc done");

	// Copy over input from host to device
	cudaMemcpy2D(da, pitch, a, rowsize, rowsize, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(db, pitch, b, rowsize, rowsize, n, cudaMemcpyHostToDevice);

	printf("Copied to device");

	dim3 blocksize(16, 16);
	dim3 gridsize((n + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
	assert(pitch % sizeof(float) == 0);
	const int ipitch = pitch / sizeof(float);
	MatAdd<<<gridsize, blocksize>>>(n, dc, da, db, ipitch);

	// Copy over output from device to host
	cudaMemcpy2D(c, rowsize, dc, pitch, rowsize, n, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

}

