/*
 Poisson solver with CG, SOR
 Compile with : g++ -fopenmp main.cpp
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "const.cpp"
#include "operator.cpp"
#include "init.cpp"

// constants
const double Lx   = 1.0;            	// x computational domain size
const double Ly   = 1.0;            	// y computational domain size
const int ghost = 1;            	// ghost zones
const int N = N_ln + 2*ghost;  	 	// total number of cells including ghost zones
const double u0  = 1.0;             	// background density
const double omega = 4.0/(2.0 + sqrt(4.0 - \
		(2.0*cos(M_PI/N_ln))*(2.0*cos(M_PI/N_ln)) ));
int itr = 0;				// count iteration
double bb = 0.0;			// criteria standard

// derived constants
const double dx = Lx/N_ln;          	// spatial resolution
const double dy = Ly/N_ln;

// pointer
double *u = new double[N*N];		// potential
double *d = new double[N_ln*N_ln];	// density
double *r = new double[N_ln*N_ln];	// residual vector 
double *p = new double[N*N];		// search direction
double *Ap = new double[N_ln*N_ln];	// <A|p>

// function prototypes
double SOR(double);		
double CG();

int main( int argc, char *argv[] )
{

	// initial condition
	const_bc(u,u0,N);
	point_source(d,N_ln);
	CG_init(u,d,r,p,bb,dx,dy,N,N_ln);

	// start evolution (selected method)
	double error = 1.0;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	while (error >= criteria){
		if (method == 0)
			error = SOR(omega);
		else
			error = CG(); 		

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
	
	// generate final result data for plotting
        std::ofstream ofs;
        ofs.open("output.txt");
        if (!ofs.is_open()) {
                printf("Failed to open file.\n");
        } else {
                for(int i=0; i<N; i++)
                for(int j=0; j<N; j++){
                        ofs << u[N*i+j] << " ";

                }
                ofs.close();
        }	

	if (method == 0) {
		printf("SOR Poisson Solver\n");
		printf("----------------------------------\n");
		printf("opt omega = %f\n",omega);
	}
	else{
		 printf("Conjugate Gradient Poisson Solver\n"); 
		 printf("----------------------------------\n");
	}
	printf("N = %d\n",N_ln);
	printf("CPU Parallel with %d threads.\n",Nthread);
	printf("Iteration = %d, error = %1.3e\n",itr,error);
	printf("Iteration Wallclock time : %f s \n",time);

	delete []u;
	delete []d;
	delete []r;
	delete []p;
	delete []Ap;
	
	return 0;
}


double SOR(double omega)
{
	double residual = 0.0;
	double dd = 0.0;
	omp_set_num_threads(Nthread);
#	pragma omp parallel

	{	
//	const int tid = omp_get_thread_num();
//	const int nt  = omp_get_num_threads();
		// odd loop
#		pragma omp for  
		for(int i=0;i<N_ln;i++){
			double tmp_residual = 0.0;
			double tmp_d = 0.0;
			for(int j=i%2;j<N_ln;j+=2){
				double psy = (u[N*i+(j+1)] + u[N*(i+2)+(j+1)] + u[N*(i+1)+j] +\
			      	      u[N*(i+1)+(j+2)] -4.0*u[N*(i+1)+(j+1)] - dx*dy*d[N_ln*i+j]);
				u[N*(i+1)+(j+1)] += 0.25*omega*psy;
				//tmp_residual += fabs(psy/u[N*(i+1)+(j+1)])/(N_ln*N_ln);
				tmp_residual += fabs(dx*dy*psy);
				tmp_d += fabs(d[N_ln*i+j]);
			}
#			pragma omp critical
			residual += tmp_residual;
			dd += tmp_d;
//			printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
		}

		// even loop
#		pragma omp for
		for(int i=0;i<N_ln;i++){
			double tmp_residual = 0.0;
			double tmp_d = 0.0;
			for(int j=(i+1)%2;j<N_ln;j+=2){
				double psy = (u[N*i+(j+1)] + u[N*(i+2)+(j+1)] + u[N*(i+1)+j] +\
			       	      u[N*(i+1)+(j+2)] -4.0*u[N*(i+1)+(j+1)] - dx*dy*d[N_ln*i+j]);
				u[N*(i+1)+(j+1)] += 0.25*omega*psy;
				//tmp_residual += fabs(psy/u[N*(i+1)+(j+1)])/(N_ln*N_ln);
				tmp_residual += fabs(dx*dy*psy);
				tmp_d += fabs(d[N_ln*i+j]);
			}
#                       pragma omp critical
			residual += tmp_residual;
			dd += tmp_d;
//			printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
		}

	}
	
	residual = residual/dd;

	return residual;
}

double CG()
{
	omp_set_num_threads(Nthread);
	double pAp = 0.0;	// p*A*p
	double alpha = 0.0;	// for update x (u) 
	double beta = 0.0;	// for update pk (search direction)
	double rr0 = 0.0; 	// old r*r
	double rr1 = 0.0;	// new r*r
	double err = 0.0;
		
	rr0 = inner_product(r,r,0,N,N_ln);		
	laplacian(Ap,p,dx,dy,N,N_ln);			// A.p
	pAp = inner_product(p,Ap,1,N,N_ln);		// pAp	
	alpha = rr0/pAp;	
	
	YPEAX(u,p,alpha,N);             // update u
        YPEAX(r,Ap,-alpha,N_ln);        // update r

	rr1 = inner_product(r,r,0,N,N_ln);
	beta = rr1/rr0;

	YEAYPX(p,r,beta,N,N_ln);        // update p	

	rr0 = rr1;
	//printf("err = %2.15f\n",rr1);
	
	err = sqrt(rr1/bb);
		
	return err ;
}


