/*
 Poisson solver with CG, SOR
 Before Compile : make clean
 Compile with : g++ -fopenmp main.cpp -o main
 Run with ./main --[Method] [Num of Threads]
 Exmaple : Run CG with 4 threads ./main --CG 4
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <getopt.h>
#include "constants.cpp"
#include "operator.cpp"
#include "init.cpp"
#include "sor.cpp"
#include "cg.cpp"

void writeToFile(double *u, int itr)
{
	char filename[64];
	std::ofstream ofs;

	if( itr < 10 )
	{
		sprintf(filename, "./output/output_0%d.txt", itr);
	}
	else
	{
		sprintf(filename, "./output/output_%d.txt", itr);
	}
	
	ofs.open( filename );
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
	double start, end, time; 

	while ( ( c = getopt_long( argc, argv, "s:c:n:", long_options, &myIndex) ) != -1)
	{
		if (c == 's')
		{
			optionSOR = true;
			Nthread = atoi(optarg);
		}

		if (c == 'c')
		{
			optionCG = true;
			Nthread = atoi(optarg);
		}

		if( c == 'n')
		{
			N_ln = atoi(optarg);
		}
	}

	const_bc(u, u0, N);
	point_source(d, N_ln);
	writeToFile(u, itr / 100);
	
	if (optionSOR)
	{
		printf("itr	error\n");
		printf("--------------\n");
		
		start = omp_get_wtime();
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
				writeToFile(u, itr / 100);
				printf("%d	%1.3e\n", itr, error);
			}
		}
		end = omp_get_wtime();
		time = end - start;

		writeToFile(u, itr / 100 + 1);

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
		
		CG_init(u, d, r, p, bb, dx, dy, N, N_ln);

		
		printf("itr	error\n");
		printf("--------------\n");
		
		start = omp_get_wtime();
		
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
				writeToFile(u, itr / 100);
				printf("%d	%1.3e\n", itr, error);
			}
		}
		end = omp_get_wtime();
                time = end - start;
		
		writeToFile(u, itr / 100 + 1);

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
