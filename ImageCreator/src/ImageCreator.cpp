//============================================================================
// Name        : ImageCreator.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "jbutil.h"

#include <iostream>
using namespace std;


int main(int argc, char * argv[]) {

	float rows = atof(argv[1]);
	float cols = atof(argv[2]);

	cout << "rows: " << rows << " cols: "<< cols << endl;

	//prepare an image out
	jbutil::image<int> image_out(rows, cols, 1, 255);
	jbutil::matrix<int> matrix;
	//resize
	matrix.resize(rows, cols);

	for(int row=0; row < rows; row++){
		for(int col=0; col < cols; col++){
			matrix(row, col) = (col/cols)*255;
		}
	}

	image_out.set_channel(0, matrix);

	string filename = "outputs/test_"+to_string(int(rows))+"x"+to_string(int(cols))+".pgm";

	// save image
	ofstream file_out(filename.c_str());
	image_out.save(file_out);

	//save textfile
	ofstream textfile;
	string filenametest = "outputs/test_"+to_string(int(rows))+"x"+to_string(int(cols))+".txt";
	textfile.open(filenametest);

	textfile << to_string(int(rows)) << " " << to_string(int(cols)) << endl;
	for(int row = 0; row < int(rows); row++){
		for(int col = 0; col < int(cols); col++){
			textfile << matrix(row, col) << " ";
		}
		textfile << endl;
	}
}
