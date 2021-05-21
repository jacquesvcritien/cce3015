//============================================================================
// Name        : jvel0131-Part1Assignment
// Author      : Jacques Vella Critien
//============================================================================

#include <iostream>
using namespace std;
#include "boost/multi_array.hpp"
#include <cassert>
#include <fstream>
#include <sstream>
#include <regex>
#include "jbutil.h"

//function to get index for pointer
int getIndex(int row, int col, int cols){
	return (row * cols) + col;
}

//function to calculate integral image
void calculateIntegralImage(boost::multi_array<double, 2> arr, double *ii, int rows, int cols){

	//count columns first (left to right)
	for(int row=0; row < rows; row++){
		for(int col=0; col < cols; col++){

			int index = getIndex(row, col, cols);
			double prev = col == 0 ? 0 : ii[index-1];

			ii[index] = prev + arr[row][col];
		}
	}

	//count rows (top to bottom)
	for(int col=0; col < cols; col++){
		for(int row=0; row < rows; row++){
			int index = getIndex(row, col, cols);
			double prev = row == 0 ? 0 : ii[index-cols];
			ii[index] = prev + ii[index];

		}
	}
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

//function to save output
void saveOutput(double* ii, int rows, int cols, string filename, double t){

	//open output stream
	ofstream outputFile;
	//get filename to save
	filename = filename.substr(filename.find_last_of("/") + 1);
	filename = filename.substr(0, filename.size()-4);
	filename = filename+".txt";
	string filename_to_save = "outputs/output_"+filename;
	outputFile.open(filename_to_save);

	//output contents to file
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			int index = getIndex(row, col, cols);
			outputFile << ii[index] << " ";
		}
		outputFile << endl;
	}

	outputFile << "Time taken: " << t << "s" << endl;

	cout << "Result written to file" << endl;

	outputFile.close();
}

int main (int argc, char *argv[]) {

	//check that file was passed
	if(argc < 2 ){
		cout << "Please pass in a filename" << endl;
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

	//check if command line argument to save output to a file is passed
	bool save = true;

	if(argc == 3){
		save = argv[2] == "true" || argv[2] == "t";
	}

	//read file
	jbutil::image<int> image_in;
	std::ifstream file_in(filename.c_str());
	image_in.load(file_in);

	//init boost array
	typedef boost::multi_array<double, 2> array_type;
	int rows = image_in.get_rows();
	int cols = image_in.get_cols();
	array_type A(boost::extents[rows][cols]);
	array_type ii(boost::extents[rows][cols]);

	//fill boost arrat
	for(int row=0; row < rows; row++){
		for (int col=0; col < cols; col++){
			A[row][col] = image_in(0, row, col);
		}
	}

	// start timer
	double t = jbutil::gettime();

	//calculate values
	calculateIntegralImage(A, ii.data(), rows, cols);

	// stop timer
	t = jbutil::gettime() - t;

	//if to save output
	if(save){
		saveOutput(ii.data(), rows, cols, filename, t);
	}

	std::cerr << "Time taken: " << t << "s" << std::endl;

	return 0;
}
