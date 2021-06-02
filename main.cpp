/*
 Poisson solver with CG, SOR
 Compile with : g++ -fopenmp main.cpp -o main
 Run with ./main --[Method] [Num of Threads]
 Exmaple : Run CG with 4 threads ./main --CG 4
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <getopt.h>
#include "constants.cpp"
#include "operator.cpp"
#include "init.cpp"
#include "sor.cpp"
#include "cg.cpp"

void writeToFile(double *u)
{
	std::ofstream ofs;
	ofs.open("output.txt");
	if (!ofs.is_open())
	{
		printf("Failed to open file.\n");
	}
	else
	{
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				ofs << u[N * i + j] << " ";
			}
		ofs.close();
	}
}

int main(int argc, char *argv[])
{
	int c = 0, myIndex = 0;
	bool optionSOR = false, optionCG = false;
	double error = 1.0;
	struct timespec start, end;

	while ( ( c = getopt_long( argc, argv, "s:c:", long_options, &myIndex) ) != -1)
	{
		if (c == 's')
		{
			optionSOR = true;
			Nthread = atoi(optarg);
			break;
		}

		if (c == 'c')
		{
			optionCG = true;
			Nthread = atoi(optarg);
			break;
		}
	}

	if (optionSOR)
	{
		const_bc(u, u0, N);
		point_source(d, N_ln);
		printf("itr	error\n");
		printf("--------------\n");
		
		clock_gettime(CLOCK_REALTIME, &start);

		while (error > criteria)
		{
			error = SOR(omega);
			itr++;
			if (itr >= 20000)
			{
				printf("Convergence Failure.\n");
				break;
			}

			if (itr % 100 == 0)
			{
				printf("%d	%1.3e\n", itr, error);
			}
		}
		clock_gettime(CLOCK_REALTIME, &end);

		// calculate wallclock time
		long seconds = end.tv_sec - start.tv_sec;
		long nanoseconds = end.tv_nsec - start.tv_nsec;
		double time = seconds + nanoseconds * 1e-9;

		writeToFile(u);

		printf("\nSOR Poisson Solver\n");
		printf("----------------------------------\n");
		printf("opt omega = %f\n", omega);
		printf("N = %d\n", N_ln);
		printf("CPU Parallel with %d threads.\n", Nthread);
		printf("Iteration = %d, error = %1.3e\n", itr, error);
		printf("Iteration Wallclock time : %f s \n", time);
	}

	if (optionCG)
	{
		const_bc(u, u0, N);
		point_source(d, N_ln);
		CG_init(u, d, r, p, bb, dx, dy, N, N_ln);
		
		printf("itr	error\n");
		printf("--------------\n");
		clock_gettime(CLOCK_REALTIME, &start);

		while (error > criteria)
		{
			error = CG();
			itr++;
			if (itr >= 20000)
			{
				printf("Convergence Failure.\n");
				break;
			}

			if (itr % 100 == 0)
			{
				printf("%d	%1.3e\n", itr, error);
			}
		}
		clock_gettime(CLOCK_REALTIME, &end);

		// calculate wallclock time
		long seconds = end.tv_sec - start.tv_sec;
		long nanoseconds = end.tv_nsec - start.tv_nsec;
		double time = seconds + nanoseconds * 1e-9;

		writeToFile(u);

		printf("\nConjugate Gradient Poisson Solver\n");
		printf("----------------------------------\n");
		printf("N = %d\n", N_ln);
		printf("CPU Parallel with %d threads.\n", Nthread);
		printf("Iteration = %d, error = %1.3e\n", itr, error);
		printf("Iteration Wallclock time : %f s \n", time);
	}
	
	delete[] u;
	delete[] d;
	delete[] r;
	delete[] p;
	delete[] Ap;

	return 0;
}
