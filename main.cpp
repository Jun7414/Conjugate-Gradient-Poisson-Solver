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
#include "constants.cpp"
#include "operator.cpp"
#include "init.cpp"
#include "sor.cpp"
#include "cg.cpp"

int main( int argc, char *argv[] )
{
	// initial condition
	const_bc(u,u0,N);
	point_source(d,N_ln);
	CG_init(u,d,r,p,bb,dx,dy,N,N_ln);

	printf("itr	error\n");
	printf("--------------\n");

	// start evolution (selected method)
	double error = 1.0;
	//double start, end, time;
	//start = omp_get_wtime();
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

		if (itr%100 ==0 )
		printf("%d	%1.3e\n",itr,error);
	}
	//end = omp_get_wtime();
	//time = end - start;
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
		printf("\nSOR Poisson Solver\n");
		printf("----------------------------------\n");
		printf("opt omega = %f\n",omega);
	}
	else{
		 printf("\nConjugate Gradient Poisson Solver\n");
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
