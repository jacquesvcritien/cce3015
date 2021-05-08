
using namespace std;
#include <cassert>
#include <fstream>
#include <sstream>
#include "stdio.h"
#include "jbutil.h"

//function to check if a string is a number
bool isNumber(string number)
{
    int i = 0;
    //flag for finding a '.'
    bool point = false;

    //for each character in string
    for (; number[i] != 0; i++)
    {
	//if '.'
	if(number[i] == '.'){
		//if '.' and already found '.' or is first character
		if(point || i==0){
			return false;
		}

		//set flag
		point = true;
	}
	else{
		if(!isdigit(number[i]))
			return false;
	}

    }
    return true;
}



__global__ void calculateIntegralImage(int rows, int cols, float *ii, const float *a, int pitch)
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

int main(int argc, char *argv[])
{
	//check that file was passed
	if(argc < 2 ){
		printf("Please pass in a filename\n");
		return 1;
	}

	//get filename
	string filename = argv[1];
	//open file
	ifstream file (filename);

	//if not open
	if (!file.is_open())
	{
		printf("File not found\n");
		return 1;
	}

	string line;
	//read first line
	getline (file,line);

	//read word by word in line
	istringstream iss(line);
	string arg;

	//get rows
	iss >> arg;
	if(!isNumber(arg)){
		printf("Rows must be a correct number\n");
		return 1;
	}
	int const rows = stoi(arg);

	//get cols
	iss >> arg;
	if(!isNumber(arg)){
		printf("Columns must be a correct number\n");
		return 1;
	}
	int const cols = stoi(arg);


	float a[rows][cols], ii[rows][cols];

	//read every line
	int row_counter =0;
	while ( getline (file,line) )
	{
		istringstream iss(line);
		int col_counter =0;
		//read every value in each line
		while(iss >> arg)
		{
			// check if passed value is number
			if(!isNumber(arg)){
				printf("Cell values must be valid numbers\n");
				return 1;
			}

			a[row_counter][col_counter] = stof(arg);
			ii[row_counter][col_counter] = 0;
			col_counter++;
		}

		//if not enough cols
		if(col_counter != cols){
			printf("Not all cell values were specified - columns\n");
			return 1;
		}
		row_counter++;
	}

	//if not enough rows
	if(row_counter != rows){
		printf("Not all cell values were specified - rows\n");
		return 1;
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

	// start timer
	double t = jbutil::gettime();	

	// Copy over input from host to device
	cudaMemcpy2D(da, pitch, a, rowsize, rowsize, rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dii, pitch, ii, rowsize, rowsize, rows, cudaMemcpyHostToDevice);

	int threadsInBlocks = 64;
	const int nblocks = (rows + (threadsInBlocks-1)) / threadsInBlocks;
	assert(pitch % sizeof(float) == 0);
	const int ipitch = pitch / sizeof(float);
	calculateIntegralImage<<<nblocks, 64>>>(rows, cols, dii, da, ipitch);

	// Copy over output from device to host
	cudaMemcpy2D(ii, rowsize, dii, pitch, rowsize, rows, cudaMemcpyDeviceToHost);

	// stop timer
	t = jbutil::gettime() - t;

	printf("\nOUTPUT\n");
	 for(int i=0; i < rows;i++){
                for(int j=0; j < cols;j++){
                        printf("%f ", ii[i][j]);
                }
                printf("\n");
        }

	printf("Time taken: %fs\n", t);

	// Free device memory
	cudaFree(da);
	cudaFree(dii);

}
