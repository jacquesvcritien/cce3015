
using namespace std;
#include <cassert>
#include <array>
#include <fstream>
#include <sstream>
#include "stdio.h"
#include "jbutil.h"

void saveOutput(float *ii, int rows, int cols, string filename){

	ofstream outputFile;
	filename = filename.substr(filename.find("/") + 1); 
	printf("%s\n", filename.c_str());
	string filename_to_save = "outputs/vectoroutput_"+filename;
	outputFile.open(filename_to_save);

	for(size_t row = 0; row < rows; row++){
		for(size_t col = 0; col < cols; col++){
			outputFile << ii[row * cols +col] << " ";
		}
		outputFile << endl;
	}

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

__global__ void calculateColumnSums(int rows, int cols, float *ii, const float *a)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < rows){
		for(int j=0; j < cols; j++){
			int index = i * cols + j;
			float prev_val = (j==0) ? 0 : ii[index-1];
			ii[index] = prev_val + a[index];
		}
	}
}

__global__ void calculateRowSums(int rows, int cols, float *ii, const float *a)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < cols){
                for(int j=0; j < rows; j++){
                        int index = j * cols + i;
                        int prev_index = index - cols;
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
	bool save = true;
	
	if(argc == 3){
		save = argv[2] == "true" || argv[2] == "t";
	}
	
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

	const int size = rows * cols * sizeof(float);
	float *a, *ii;
	a=(float*)malloc(size);	
	ii=(float*)malloc(size);	

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

			a[row_counter * cols +  col_counter] = stof(arg);
			ii[row_counter * cols + col_counter] = 0;
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

	float *da, *dii;

	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&dii, size);

	// start timer
	double t = jbutil::gettime();	

	// Copy over input from host to device
	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dii, ii, size, cudaMemcpyHostToDevice);

	free(a);

	int threadsInBlocks = 128;
	const int nblocks = (rows + (threadsInBlocks-1)) / threadsInBlocks;
	calculateColumnSums<<<nblocks, 64>>>(rows, cols, dii, da);
	calculateRowSums<<<nblocks, 64>>>(rows, cols, dii, da);

	// Copy over output from device to host
	cudaMemcpy(ii, dii, size, cudaMemcpyDeviceToHost);

	// stop timer
	t = jbutil::gettime() - t;

	printf("Time taken: %fs\n", t);

	if(save){
		saveOutput(ii, rows, cols, filename);
	}

	free(ii);
	// Free device memory
	cudaFree(da);
	cudaFree(dii);

}
