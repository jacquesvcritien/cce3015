

#include <cassert>
#include "stdio.h"

__global__ void VecAdd(int rows, int cols, float *ii, const float *a, int pitch)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < rows){
		for(int j=0; j < cols; j++){
			int index = i * pitch + j;
			float prev_val = (j==0) ? 0 : ii[index-1];
			ii[index] = prev_val + a[index];
		}
	}

	__syncthreads();


	if(i < cols){
                for(int j=0; j < rows; j++){
                        int index = j * pitch + i;
                        int prev_index = (j-1) * pitch + i;
                        float prev_val = (j==0) ? 0 : ii[prev_index];
                        ii[index] = prev_val + ii[index];
                }
        }
}

int main()
{
	const int rows = 4;
	const int cols = 4;
	float a[rows][cols], ii[rows][cols];

	for(int i=0; i < cols; i++){
		for(int j=0; j < rows; j++){
			a[j][i] = i+1;
			ii[j][i] = 0;
		}
	}

	printf("INPUT\n");
	 for(int i=0; i < rows;i++){
                for(int j=0; j < cols;j++){
                        printf("%f ", a[i][j]);
                }
                printf("\n");
        }

	const int rowsize = cols * sizeof(float);
	float *da, *dii;

	size_t pitch;
	cudaMallocPitch((void**)&da, &pitch, rowsize, rows);
	cudaMallocPitch((void**)&dii, &pitch, rowsize, rows);

	// Copy over input from host to device
	cudaMemcpy2D(da, pitch, a, rowsize, rowsize, rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dii, pitch, ii, rowsize, rowsize, rows, cudaMemcpyHostToDevice);

	int threadsInBlocks = 64;
	const int nblocks = (rows + (threadsInBlocks-1)) / threadsInBlocks;
	assert(pitch % sizeof(float) == 0);
	const int ipitch = pitch / sizeof(float);
	VecAdd<<<nblocks, 64>>>(rows, cols, dii, da, ipitch);

	// Copy over output from device to host
	cudaMemcpy2D(ii, rowsize, dii, pitch, rowsize, rows, cudaMemcpyDeviceToHost);

	printf("\nOUTPUT\n");
	 for(int i=0; i < rows;i++){
                for(int j=0; j < cols;j++){
                        printf("%f ", ii[i][j]);
                }
                printf("\n");
        }

	// Free device memory
	cudaFree(da);
	cudaFree(dii);

}
