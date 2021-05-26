/**
 * What was improved?
 * - Removed malloc and changed them to cudaAllocHost
 * - Added cudaHostAllocWriteCombined flag
 */

using namespace std;
#include <cassert>
#include <array>
#include <fstream>
#include <sstream>
#include "stdio.h"
#include "jbutil.h"

//function to save output
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

//function to calculate row cumulative sums
__global__ void cumulativeRowPass(int rows, int cols, float *ii)
{
	//get row
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < rows){
		//get row index
		int row_index = i * cols;
		//for each column
		for(int j=0; j < cols; j++){
			//get index from array
			int index = row_index + j;
			//get previous value
			int prev_val = (j==0) ? 0 : ii[index-1];
			ii[index] = prev_val + ii[index];
		}
	}
}

//function to calculate column cumulative sums
__global__ void cumulativeColumnPass(int rows, int cols, float *ii)
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
	//allocate memory to host with write combined flag
	cudaHostAlloc((void**)&ii, size, cudaHostAllocWriteCombined);

	//fill array from image
	for(int row=0; row < rows; row++){
		for (int col=0; col < cols; col++){
			ii[row * cols + col] = image_in(0, row, col);
		}
	}

	float* dii;
	cudaMalloc((void**)&dii, size);

	double totalTime = 0;
	// start timer
	double t = jbutil::gettime();

	// Copy over input from host to device
	cudaMemcpy(dii, ii, size, cudaMemcpyHostToDevice);

	// stop timer
	t = jbutil::gettime() - t;
	printf("Time taken to copy from host to device: %fs\n", t);
	totalTime += t;


	int threadsInBlocks = 128;
	const int nblocks = (rows + (threadsInBlocks-1)) / threadsInBlocks;
	printf("Number of threads in blocks: %d\n", threadsInBlocks);
	printf("Number of blocks: %d\n", nblocks);

	// start timer
	t = jbutil::gettime();

	//start kernels
	cumulativeRowPass<<<nblocks, threadsInBlocks>>>(rows, cols, dii);
	cumulativeColumnPass<<<nblocks, threadsInBlocks>>>(rows, cols, dii);

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

	cudaFreeHost(ii);
	//free device memory
	cudaFree(dii);

}
