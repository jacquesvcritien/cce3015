//============================================================================
// Name        : ArgumentGenerator.cpp
// Author      : Jacques Vella Critien
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;
#include "boost/multi_array.hpp"
#include <fstream>
int main() {
	int rows = 100;
	int cols = 100;

	//init array
//	double *arr[rows];

	typedef boost::multi_array<double, 2> array_type;
	array_type A(boost::extents[rows][cols]);

//	for(int j=0; j < rows; j++){
//		arr[j] = new double[cols];
//	}


	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			A[row][col] = col+1;
		}
	}

	//print input
//	cout << "INPUT" << endl;
//	for(int row = 0; row < rows; row++){
//		for(int col = 0; col < cols; col++){
//			cout << arr[row][col] << " ";
//		}
//		cout << endl;
//	}

	//print input
	cout << "[";
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			cout << A[row][col] << " ";
		}
		cout << ';';
	}


	cout << ']' << endl;

	ofstream myfile;
	string filename = "example"+to_string(rows)+"x"+to_string(cols)+".txt";
	myfile.open(filename);


	myfile << rows << " " << cols << endl;
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			myfile << A[row][col] << " ";
		}
		myfile << endl;
	}

	cout << "File created" << endl;

	  myfile.close();
	  return 0;

}
