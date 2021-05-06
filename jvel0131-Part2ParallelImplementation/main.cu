#include <cassert>

__global__ void VecAdd(int n, float *ii, const float *a, int cols)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		for(int j=0; j < 3; j++){
			int index = i * cols + j;
			ii[index] = a[index];
		}
	}
}

int main()
{
	const int n = 3;
	float a[n][n], ii[n][n];

	for(int i=0; i < n; i++){
		for(int j=0; j < n; j++){
			a[i][j] = i;
			ii[i][j] = 0;
		}
	}

	const int rowsize = n * sizeof(float);
	float *da, *dii;

	size_t pitch;
	cudaMallocPitch((void**)&da, &pitch, rowsize, n);
	cudaMallocPitch((void**)&dii, &pitch, rowsize, n);

	// Copy over input from host to device
	cudaMemcpy2D(da, pitch, a, rowsize, rowsize, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dii, pitch, ii, rowsize, rowsize, n, cudaMemcpyHostToDevice);

//	dim3 blocksize(16, 16);
//	dim3 gridsize((n + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
//	assert(pitch % sizeof(float) == 0);
//	const int ipitch = pitch / sizeof(float);
//	MatAdd<<<gridsize, blocksize>>>(n, dc, da, db, ipitch);

	const int nblocks = (n + 63) / 64;
	VecAdd<<<nblocks, 64>>>(n, dii, da, 3);

	// Copy over output from device to host
	cudaMemcpy2D(ii, rowsize, dii, pitch, rowsize, n, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(da);
	cudaFree(dii);

}
