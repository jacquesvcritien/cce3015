/**
 * What was improved?
 * - Removed the row pass kernel and kept only the column pass kernel
 * - Implemented a transpose kernel which copies the array to another one in a transposed way
 * - This was done to reduce time taken due to uncoalesced memory
 */

using namespace std;
#include <cassert>
#include <array>
#include <fstream>
#include <sstream>
#include "stdio.h"
#include "jbutil.h"

//function to save output to file
void saveOutput(float *ii, int rows, int cols, string filename, double t){

	ofstream outputFile;
	filename = filename.substr(filename.find_last_of("/") + 1);
	filename = filename.substr(0, filename.size()-4);
	filename = filename+".txt";
	string filename_to_save = "outputs/output_"+filename;
	outputFile.open(filename_to_save);

	for(size_t row = 0; row < rows; row++){
		for(size_t col = 0; col < cols; col++){
			outputFile << ii[row * cols +col] << " ";
		}
		outputFile << endl;
	}

	outputFile << "Time taken: " << t << "s" << endl;

	cout << "Result written to file" << endl;

	outputFile.close();
}

//kernel to transpose - copy the array to another one
__global__ void TransposeMatrix(int rows, int cols, float *ii, float *transpose)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//get destination
	int ij = i * rows + j;
	//get source
	int ji = j * cols + i;
	if(i < cols && j < rows){
		//fill transpose array
		transpose[ij] = ii[ji];
	}
}

//kernel to calculate cumulative sums - top to bottom
__global__ void cumulativePass(int rows, int cols, float *ii)
{
	//get column index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < cols){
		//for each row in column
		for(int j=0; j < rows; j++){
			//get index from array
			int index = j * cols + i;
			//get previous index
			int prev_index = index - cols;
			//get previous value
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
	//get extension
	string ext = filename.substr(filename.size()-4, filename.size());

	if(ext != ".pgm"){
		cout << "Input must be a .pgm file" << endl;
		return 1;
	}

	//check cli argument to see whether to save output to file
	bool save = true;
	
	if(argc == 3){
		save = argv[2] == "true" || argv[2] == "t";
	}
	
	//read file
	jbutil::image<int> image_in;
	std::ifstream file_in(filename.c_str());
	image_in.load(file_in);

	//get rows and cols
	int rows = image_in.get_rows();
	int cols = image_in.get_cols();

	const int size = rows * cols * sizeof(float);
	//initialise arrays
	float *ii;
	cudaHostAlloc((void**)&ii, size, cudaHostAllocWriteCombined);

	//fill array from image
	for(int row=0; row < rows; row++){
		for (int col=0; col < cols; col++){
			ii[row * cols + col] = image_in(0, row, col);
		}
	}


	float* dii;
	float* transpose;
	//Allocate device memory for the 2 arrays
	cudaMalloc((void**)&dii, size);
	cudaMalloc((void**)&transpose, size);

	double totalTime = 0;
	// start timer
	double t = jbutil::gettime();

	// Copy over input from host to device
	cudaMemcpy(dii, ii, size, cudaMemcpyHostToDevice);

	// stop timer
	t = jbutil::gettime() - t;
	printf("Time taken to copy from host to device: %fs\n", t);
	totalTime += t;

	//determine structure for blocks and grids for the kernels
	int threadsInBlocks = 128;
	const int nblocks = (rows + (threadsInBlocks-1)) / threadsInBlocks;
	dim3 blocksize(16, 128);
	dim3 gridsize( (cols + blocksize.x - 1) / blocksize.x, (rows + blocksize.y - 1) / blocksize.y);
	printf("Number of threads in blocks for cumulative pass: %d\n", threadsInBlocks);
	printf("Number of blocks for cumulative pass: %d\n", nblocks);
	printf("Block size for transpose copy: %d x %d\n", blocksize.x, blocksize.y);
	printf("Grid size for transpose copy: %d x %d\n", gridsize.x, gridsize.y);

	// start timer
	t = jbutil::gettime();

	//start kernels
	cumulativePass<<<nblocks, threadsInBlocks>>>(rows, cols, dii);
	TransposeMatrix<<<gridsize, blocksize>>>(rows, cols, dii, transpose);
	cumulativePass<<<nblocks, threadsInBlocks>>>(cols, rows, transpose);
	TransposeMatrix<<<gridsize, blocksize>>>(cols, rows, transpose, dii);

	// stop timer
	t = jbutil::gettime() - t;
	printf("Time taken to calculate integral image: %fs\n", t);
	totalTime += t;

	t = jbutil::gettime();

	// Copy over output from device to host
	cudaMemcpy(ii, dii, size, cudaMemcpyDeviceToHost);

	// stop timer
	t = jbutil::gettime() - t;
	printf("Time taken to copy from device to host: %fs\n", t);
	totalTime += t;

	//output to file if save is true
	if(save){
		saveOutput(ii, rows, cols, filename, totalTime);
	}

	printf("Total time taken: %fs\n", totalTime);

	//free host memory
	cudaFreeHost(ii);
	//free device memory
	cudaFree(dii);
	cudaFree(transpose);

}
