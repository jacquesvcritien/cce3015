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
#include "jbutil.h"

//function to get index for pointer
int getIndex(int row, int col, int cols){
	return (row * cols) + col;
}

void calculateIntegralImage(boost::multi_array<double, 2> arr, double *ii, int rows, int cols){

	//count columns first (left to right)
	for(int row=0; row < rows; row++){
		for(int col=0; col < cols; col++){

			int index = getIndex(row, col, cols);
			double prev = col == 0 ? 0 : ii[index-1];

			ii[index] = prev + arr[row][col];
		}
	}

	//count rows first (top to bottom)
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


void saveOutput(double* ii, int rows, int cols, string filename){

	ofstream outputFile;
	string filename_to_save = "outputs/output_"+filename;
	outputFile.open(filename_to_save);

	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			int index = getIndex(row, col, cols);
			outputFile << ii[index] << " ";
		}
		outputFile << endl;
	}

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
	//open file
	ifstream file (filename);

	//if not open
	if (!file.is_open())
	{
		cout << "File not found" << endl;
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
		cout << "Rows must be a correct number" << endl;
		return 1;
	}
	int rows = stoi(arg);

	//get cols
	iss >> arg;
	if(!isNumber(arg)){
		cout << "Columns must be a correct number" << endl;
		return 1;
	}
	int cols = stoi(arg);

	//init boost array
	typedef boost::multi_array<double, 2> array_type;
	array_type A(boost::extents[rows][cols]);
	array_type ii(boost::extents[rows][cols]);

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
				cout << "Cell values must be valid numbers" << endl;
				return 1;
			}

			A[row_counter][col_counter] = stof(arg);
			col_counter++;
		}

		//if not enough cols
		if(col_counter != cols){
			cout << "Not all cell values were specified - columns" << endl;
			return 1;
		}
		row_counter++;
	}

	//if not enough rows
	if(row_counter != rows){
		cout << "Not all cell values were specified - rows" << endl;
		return 1;
	}

	// start timer
	double t = jbutil::gettime();

	//calculate values
	calculateIntegralImage(A, ii.data(), rows, cols);

	// stop timer
	t = jbutil::gettime() - t;

	//print output
	cout << "OUTPUT" << endl;
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			cout << ii[row][col] << " ";
		}
		cout << endl;
	}

	saveOutput(ii.data(), rows, cols, filename);

	std::cerr << "Time taken: " << t << "s" << std::endl;

	return 0;
}
