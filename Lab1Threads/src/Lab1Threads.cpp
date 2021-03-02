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
#include <pthread.h>
#include "jbutil.h"

#include <iostream>
using namespace std;

typedef struct {
	int n; // length to compute
	pthread_t tid;
	double *m;
	double *a;
	double *b;
	double *A;
	double *B;
	pthread_mutex_t *mutex;
} tdata_t;



double functionGiven(double x, int miu, int sigma) {
	return 1 / sqrt(2 * M_PI * (pow(sigma, 2)))
			* exp(-((pow(x - miu, 2) / (2 * pow(sigma, 2)))));
}

// Monte Carlo integration function
double MonteCarlo(int N, double a, double b, double A, double B) {
	std::cerr << "\nImplementation (" << N << " samples)" << std::endl;

	jbutil::randgen gen;

	int miu = 0;
	int sigma = 1;

	double m = 0;

	for (int i = 0; i < N; i++) {
		double generated_x = gen.fval(a, b);
		double generated_y = gen.fval(A, B);

		double actual_y = functionGiven(generated_x, miu, sigma);

		if (generated_y < actual_y) {
			m++;
		}

	}

	return m;


}

void* ThreadFunc(void *arg) {
	tdata_t *d = (tdata_t*) arg;
	double m = MonteCarlo(d->n, *d->a, *d->b, *d->A, *d->B);
	pthread_mutex_lock(d->mutex);
	*d->m += m;
	pthread_mutex_unlock(d->mutex);
	return NULL;
}

// Main program entry point

int main() {
	std::cerr << "Lab 1: Monte Carlo integration" << std::endl;
	const int N = int(1E8);

	// problem data
	const int NumThreads = 4;
	double m = 0;

	double a = -2;
	double b = 2;
	double A = 0;
	double B = 0.4;

	pthread_mutex_t mutex;
	// set up thread-specific data
	tdata_t d[NumThreads];
	for (int i = 0; i < NumThreads; i++) {
		d[i].n = N / NumThreads;
		d[i].m = &m;
		d[i].a = &a;
		d[i].b = &b;
		d[i].A = &A;
		d[i].B = &B;
		d[i].mutex = &mutex;
	}

	// start timer
	double t = jbutil::gettime();

	pthread_mutex_init(&mutex, NULL);
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for (int i = 0; i < NumThreads; i++) {
		if (pthread_create(&d[i].tid, &attr, ThreadFunc, (void*) &d[i]))
			failwith("pthread_create() failed.");
	}
	for (int i = 0; i < NumThreads; i++) {
		if (pthread_join(d[i].tid, NULL))
			failwith("pthread_join() failed.");
	}


	double estimate = (m / N) * (((B - A) * (b - a)) + (A * (b - a)));

	std::cerr << "estimate is " << estimate << std::endl;
	std::cerr << "error function result " << erf(sqrt(2)) << std::endl;
	// stop timer
	t = jbutil::gettime() - t;
	std::cerr << "Time taken: " << t << "s" << std::endl;

	std::cout << "Ending process\n";
	return 0;

}
