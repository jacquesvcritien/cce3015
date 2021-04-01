//============================================================================
// Name        : Assignment-Part1.cpp
// Author      : Jacques Vella Critien
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;
#include "jbutil.h"
#include <pthread.h>

double s(int col, double *arr){

	if(col == 0)
		return arr[col];

	return s(col-1, arr) + arr[col];
}

//function to check if a string is a number
bool isNumber(char number[])
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

int main(int argc, char *argv[]) {

	cout << "Integral Image Calculation" << endl;

	//check that enough arguments are passed or arguments are not numbers
	if(argc < 3 || !isNumber(argv[1]) || !isNumber(argv[2])){
		cout << "Please pass in the number of rows and columns" << endl;
		return 1;
	}

	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);

	//init array
	double *arr[rows];
	double *ii[rows+1];

	for(int j=0; j < rows; j++){
		arr[j] = new double[cols];
		ii[j] = new double[cols+1];
		ii[j][0] = 0;

		if(j == rows - 1){
			ii[j+1] = new double[cols+1];
			ii[j+1][0] = 0;
		}
	}

	int expectedArgs = 3 + (rows*cols);

	//check that enough arguments are passed
	if(argc < expectedArgs){
		cout << "Not enough arguments are passed" << endl;
		return 1;
	}

	int counter = 0;
	//read cli arguments to arrays
	for(int row=0; row<rows; row++){
		for(int col=0; col<cols; col++){

			//get index of argument
			int argIndex = 3+counter;

			// check if passed argument is number
			if(!isNumber(argv[argIndex])){
				cout << "Arguments must be valid numbers" << endl;
				return 1;
			}

			arr[row][col] = atof(argv[argIndex]);
			counter++;
		}
	}

	//print input
	cout << "INPUT" << endl;
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			cout << arr[row][col] << " ";
		}
		cout << endl;
	}

	// start timer
	double t = jbutil::gettime();

	//perform calculation
	for(int row=0; row < rows; row++){
		for(int col=0; col < cols; col++){
			double sum = s(col, arr[row]);
			ii[row+1][col+1] = ii[row][col+1] + sum;
		}
	}

	// stop timer
	t = jbutil::gettime() - t;
	cout << endl;


	//print output
	cout << "OUTPUT" << endl;
	for(int row = 1; row < rows + 1; row++){
		for(int col = 1; col < cols + 1; col++){
			cout << ii[row][col] << " ";
		}
		cout << endl;
	}


	std::cerr << "Time taken: " << t << "s" << std::endl;

	return 0;
}
