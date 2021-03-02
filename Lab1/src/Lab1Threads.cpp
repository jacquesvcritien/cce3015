/*!
 * \file
 * \brief   Lab 1 - Threading.
 * \author  Johann Briffa
 *
 * Template for the solution to Lab 1 practical exercise on Monte Carlo
 * integration.
 *
 * \section svn Version Control
 */

#include <math.h>
#include <pthreads.h>
#include "jbutil.h"

#include <iostream>
using namespace std;

typedef struct {
	int n; // length to compute
	pthread_t tid;
	double *s; // shared sum
	pthread_mutex_t *mutex;
} tdata_t;

void* ThreadFunc(void *arg) {
	tdata_t *d = (tdata_t*) arg;
	int s = 0; // local sum
	for (int i = 0; i < d->n; i++)
		s += d->a[i] * d->b[i];
	pthread_mutex_lock(d->mutex);
	std::cerr << "Thread " << d->tid << " storing result" << std::endl;
	*d->s += s;
	pthread_mutex_unlock(d->mutex);
	return NULL;
}

double functionGiven(double x, int miu, int sigma){
	return 1/sqrt(2* M_PI *(pow(sigma, 2))) * exp(-((pow(x - miu, 2)/(2*pow(sigma,2)))))  ;
}



// Monte Carlo integration function

void MonteCarlo(const int N)
   {
   std::cerr << "\nImplementation (" << N << " samples)" << std::endl;
   // start timer
   double t = jbutil::gettime();

   jbutil::randgen gen;

   int miu = 0;
   int sigma = 1;


   double a = -2;
   double b = 2;
   double A = 0;
   double B = 0.4;

   double m = 0;

   for(int i=0; i < N; i++){
	   double generated_x = gen.fval(a, b);
	   double generated_y = gen.fval(A, B);

	   double actual_y = functionGiven(generated_x, miu, sigma);

	   if(generated_y < actual_y){
		   m++;
	   }

   }

   double estimate = (m/N) * (((B - A) * (b - a)) + (A * (b - a)));

   std::cerr << "estimate is " << estimate << std::endl;

   // stop timer
   t = jbutil::gettime() - t;
   std::cerr << "Time taken: " << t << "s" << std::endl;
   }

// Main program entry point

int main()
{
	std::cerr << "Lab 1: Monte Carlo integration" << std::endl;
	const int N = int(1E8);
	MonteCarlo(N);

	std::cerr << "error function result " << erf(sqrt(2)) << std::endl;
}
