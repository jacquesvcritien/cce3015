/*!
 * \file
 * \brief   Lab 2 - SIMD Programming.
 * \author  Johann Briffa
 *
 * Template for the solution to Lab 2 practical exercise on image resampling
 * using the Lanczos filter.
 *
 * \section svn Version Control
 */

#include "jbutil.h"

#include <cmath>
#include <math.h>

#include <iostream>
using namespace std;

// Resample the image using Lanczos filter

//sinc(x) function
double sinc(double x){
	if(x == 0){
		return 1;
	}
	else{
		double pix = M_PI * x;
		return sin(pix)/pix;
	}
}

//L(x) function
double lfunc(double x, int a){
	if((-1*a) < x && x < a){
		return sinc(x)*sinc(x/a);
	}
	else{
		return 0;
	}
}

double clip(double n) {
  return std::max(0.0, std::min(n, 255.0));
}

template <class real>
jbutil::image<int> my_func(jbutil::image<int> image_in, const real R, const int a){


	//get rows
	double rows = image_in.get_rows();
	//get cols
	double cols = image_in.get_cols();

	int output_rows = rows * R;
	int output_cols = cols * R;

	//prepare an image out
	jbutil::image<int> image_out(output_rows, output_cols, 1, 255);
	jbutil::matrix<int> matrix;
	//resize
	matrix.resize(rows, cols);

	//for each row
	for(int m = 0; m < output_rows; m++){
		//for each column
		for(int n = 0; n < output_cols; n++){

			//calculate monr
			double monr = (m/R);
			//get max i
			double max_i = floor(monr + a);
			//get i
			double i = ceil(monr - a);

			//calculate nonr
			double nonr = (n/R);
			//get max j
			double max_j = floor(nonr + a);
			//get j
			double j = ceil(nonr - a);

			//initialise sum
			double sum = 0;

			//from i up till max i
			for(; i <= max_i; i++){
				//from j up till max j
				for(; j <= max_j; j++){
					//get f
					double f = image_in(0, clip(i), clip(j));
					//calculate l1
					double lfunc1 = lfunc(monr-i, a);
					//calculate l2
					double lfunc2 = lfunc(nonr-j, a);

					//add to sum
					sum += f * lfunc1 * lfunc2;
				}
			}

			//fill matrix
			matrix(m, n) = sum;


		}
	}

	image_out.set_channel(0, matrix);
	return image_out;
}


template <class real>
void process(const std::string infile, const std::string outfile,
      const real R, const int a)
   {
   // load image
   jbutil::image<int> image_in;
   std::ifstream file_in(infile.c_str());
   image_in.load(file_in);
   // start timer
   double t = jbutil::gettime();

   // f[i,j] is image_in(0, i, j)

   // TODO: Implemented image resampling here
   jbutil::image<int> image_out = my_func(image_in, R, a);

   // stop timer
   t = jbutil::gettime() - t;
   // save image
   std::ofstream file_out(outfile.c_str());
   image_out.save(file_out);
   // show time taken
   std::cerr << "Time taken: " << t << "s" << std::endl;
   }

// Main program entry point

int main(int argc, char *argv[])
   {
   std::cerr << "Lab 2: Image resampling with Lanczos filter" << std::endl;

   std::cerr << "R: " << atof(argv[3]) << ", a:" << atoi(argv[4]) << std::endl;

   if (argc != 5)
      {
      std::cerr << "Usage: " << argv[0]
            << " <infile> <outfile> <scale-factor> <limit>" << std::endl;
      exit(1);
      }
   process<float> (argv[1], argv[2], atof(argv[3]), atoi(argv[4]));
   }
