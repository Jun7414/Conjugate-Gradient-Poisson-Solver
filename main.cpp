/*
 Parallel Computation SOR
 Compile with : g++ -fopenmp main.cpp
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <time.h>

const int Nthread = 8;

const int N_ln   = 64;                  // number of computing cell in 1D
const double Lx   = 1.0;            	// x computational domain size
const double Ly   = 1.0;            	// y computational domain size
const int ghost = 1;            	// ghost zones
const int N = N_ln + 2*ghost;  	 	// total number of cells including ghost zones
const double D   = 1.0;             	// diffusion coefficient
const double u0  = 1.0;             	// background density
const double amp = 1.0;             	// sinusoidal amplitude
const double criteria = 1.0e-14;     	// convergence criteria
const double omega = 4.0/(2.0 + sqrt(4.0 - \
		(2.0*cos(M_PI/N_ln))*(2.0*cos(M_PI/N_ln)) ));
int itr = 0;				// count iteration
// derived constants
const double dx = Lx/N_ln;          	// spatial resolution
const double dy = Ly/N_ln;
const double dt = dx*dy/(4*D);      	// CLF stability

// Pointer for potential
double *u;				// potential
double *d;				// density
double *x, *y;				// coordinates

// function prototypes
void const_bc(double *);
void point_source(double *);		// 4 grid in middle has density		
double SOR_parallel(double);		

// function for memories reference
void *memalloc(size_t n)
{
	void *v;

	if ((v = malloc(n)) == NULL) {
		printf("Not enough memory!!\n");
		exit(1);
	}
	return v;
}


int main( int argc, char *argv[] )
{

	// distribute memory
	x = (double*)memalloc(N_ln*sizeof(double));
	y = (double*)memalloc(N_ln*sizeof(double));
	u = (double*)memalloc((N*N)*sizeof(double));
	d = (double*)memalloc((N_ln*N_ln)*sizeof(double));
	
	// initial condition
	const_bc(u);
	point_source(d);

	// start evolution (selected method)
	double residual = 1.0;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);
	while (residual >= criteria){ 		
		residual = SOR_parallel(omega);
		itr ++;
		if(itr >= 20000) {
			printf("Convergence Failure.\n");
			break;
			} 
	}
	clock_gettime(CLOCK_REALTIME, &end);

	// calculate wallclock time
	long seconds = end.tv_sec - start.tv_sec;
	long nanoseconds = end.tv_nsec - start.tv_nsec;
	double time = seconds + nanoseconds*1e-9;

	printf("CPU Parallel with %d threads.\n",Nthread);
	printf("N = %d\n",N_ln);
	printf("opt omega = %f\n",omega);
	printf("Iteration = %d, residual = %1.3e\n",itr,residual);
	printf("Iteration Wallclock time : %f s \n",time);

	free(x);free(y);free(u);free(d);
	return 0;
}

void const_bc(double *u)
{
#	pragma omp parallel for collapse(2)	
	for (int i=0;i<N;i++){
		for (int j=0;j<N;j++){
			u[N*i+j] = u0;
		}
	}
}

void point_source(double *d)
{
	const double G = 1.0;
#       pragma omp parallel for collapse(2)	
	for(int i=N_ln/2-1;i <= N_ln/2;i++){
		for(int j=N_ln/2-1;j <= N_ln/2;j++){
			d[N_ln*i+j] = 4*M_PI*G*25.0;
		}
	}
}


double SOR_parallel(double omega)
{
	double residual = 0.0;
	omp_set_num_threads(Nthread);
#	pragma omp parallel
	{	
//	const int tid = omp_get_thread_num();
//	const int nt  = omp_get_num_threads();
		// odd loop
#		pragma omp for  
		for(int i=0;i<N_ln;i++){
			double tmp_residual = 0.0;
			for(int j=i%2;j<N_ln;j+=2){
				double psy = (u[N*i+(j+1)] + u[N*(i+2)+(j+1)] + u[N*(i+1)+j] +\
			      	      u[N*(i+1)+(j+2)] -4.0*u[N*(i+1)+(j+1)] - dx*dy*d[N_ln*i+j]);
				u[N*(i+1)+(j+1)] += 0.25*omega*psy;
				tmp_residual += fabs(psy/u[N*(i+1)+(j+1)])/(N_ln*N_ln);
			}
#			pragma omp critical
			residual += tmp_residual;
//			printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
		}

		// even loop
#		pragma omp for
		for(int i=0;i<N_ln;i++){
			double tmp_residual = 0.0;
			for(int j=(i+1)%2;j<N_ln;j+=2){
				double psy = (u[N*i+(j+1)] + u[N*(i+2)+(j+1)] + u[N*(i+1)+j] +\
			       	      u[N*(i+1)+(j+2)] -4.0*u[N*(i+1)+(j+1)] - dx*dy*d[N_ln*i+j]);
				u[N*(i+1)+(j+1)] += 0.25*omega*psy;
				tmp_residual += fabs(psy/u[N*(i+1)+(j+1)])/(N_ln*N_ln);
			}
#                       pragma omp critical
			residual += tmp_residual;
//			printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
		}

	}

	return residual;
}
