//============================================================================
// Name        : ArgumentGenerator.cpp
// Author      : Jacques Vella Critien
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int main() {
	int rows = 500;
	int cols = 500;

	//init array
	double *arr[rows];

	for(int j=0; j < rows; j++){
		arr[j] = new double[cols];
	}


	for(int row = 0; row <= rows; row++){
		for(int col = 0; col <= cols; col++){
			arr[row][col] = col+1;
		}
	}

	//print input
//	cout << "INPUT" << endl;
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
//			cout << arr[row][col] << " ";
		}
//		cout << endl;
	}

	//print input
//	cout << "[";
//	for(int row = 0; row < rows; row++){
//		for(int col = 0; col < cols; col++){
//			cout << arr[row][col] << " ";
//		}
//		cout << ';';
//	}
//
//
//	cout << ']' << endl;

	cout << rows << " " << cols << " ";
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			cout << arr[row][col] << " ";
		}
	}
	cout << endl;


}
