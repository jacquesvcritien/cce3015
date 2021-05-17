
using namespace std;
#include <cassert>
#include <array>
#include <fstream>
#include <sstream>
#include "stdio.h"
#include "jbutil.h"

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

//function to calculate row cumulative sums
__global__ void cumulativeRowPass(int rows, int cols, float *ii, const float *a)
{
	//get row
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < rows){
		//for each column
		for(int j=0; j < cols; j++){
			//get index from array
			int index = i * cols + j;
			//get previous value
			float prev_val = (j==0) ? 0 : ii[index-1];
			ii[index] = prev_val + a[index];
		}
	}
}

//function to calculate column cumulative sums
__global__ void cumulativeColumnPass(int rows, int cols, float *ii, const float *a)
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
	float *a, *ii;
	a=(float*)malloc(size);	
	ii=(float*)malloc(size);	

	//fill array from image
	for(int row=0; row < rows; row++){
		for (int col=0; col < cols; col++){
			a[row * cols +  col] = image_in(0, row, col);
			ii[row * cols + col] = 0;
		}
	}

	float *da, *dii;

	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&dii, size);

	// start timer
	double t = jbutil::gettime();	

	// Copy over input from host to device
	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dii, ii, size, cudaMemcpyHostToDevice);

	//free memory for original array
	free(a);

	int threadsInBlocks = 512;
	const int nblocks = (rows + (threadsInBlocks-1)) / threadsInBlocks;
	//start kernels
	cumulativeRowPass<<<nblocks, 64>>>(rows, cols, dii, da);
	cumulativeColumnPass<<<nblocks, 64>>>(rows, cols, dii, da);

	// Copy over output from device to host
	cudaMemcpy(ii, dii, size, cudaMemcpyDeviceToHost);

	// stop timer
	t = jbutil::gettime() - t;

	printf("Time taken: %fs\n", t);

	//output to file if save is true
	if(save){
		saveOutput(ii, rows, cols, filename, t);
	}

	free(ii);
	//free device memory
	cudaFree(da);
	cudaFree(dii);

}
