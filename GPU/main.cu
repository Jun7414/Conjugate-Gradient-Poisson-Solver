/*
 Poisson solver with CG, SOR
 Compile with : g++ -fopenmp main.cpp
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "const.cu"
#include "operator.cu"
#include "init.cu"
#include "WriteToFile.cu"

// constants
const double Lx = 1.0;			// x computational domain size
const double Ly = 1.0;			// y computational domain size
const int ghost = 1;			// ghost zones
const int N = N_ln + 2 * ghost; // total number of cells including ghost zones
const double u0 = 1.0;			// background density
const double omega = 4.0 / (2.0 + sqrt(4.0 -
									   (2.0 * cos(M_PI / N_ln)) * (2.0 * cos(M_PI / N_ln))));
int itr = 0;	 // count iteration
double bb = 0.0; // criteria standard

//GPU constants
const int NThread_Per_Block = 128;
const int NBlock = (N * N + NThread_Per_Block - 1) / NThread_Per_Block;

const int BLOCK_SIZE = 8;
const int GRID_SIZE_X = (N_ln + BLOCK_SIZE - 1) / BLOCK_SIZE;
const int GRID_SIZE_Y = (N_ln + BLOCK_SIZE - 1) / BLOCK_SIZE;
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 dimGrid(GRID_SIZE_X, GRID_SIZE_Y);

// derived constants
const double dx = Lx / N_ln; // spatial resolution
const double dy = Ly / N_ln;

// pointer
double *u = new double[N * N];		  // potential
double *d = new double[N_ln * N_ln];  // density
double *r = new double[N_ln * N_ln];  // residual vector
double *p = new double[N * N];		  // search direction
double *Ap = new double[N_ln * N_ln]; // <A|p>

// function prototypes
double SOR(double);
double CG(double *, double *, double *, double *);

int main(int argc, char *argv[])
{

	// initial condition
	if (bc == 0)
		const_bc(u, u0, N);
	else if (bc == 1)
		oneside_bc(u, u0, N);
	else if (bc == 2)
		fourside_bc(u, u0, N);
	else if (bc == 3)
		sin_bc(u, u0, N);
	else
		printf("Undefined boundary condition.");

	if (source == 0)
		background_density(d, N_ln);
	else if (source == 1)
		point_source_middle(d, N_ln);
	else if (source == 2)
		point_source_4q(d, N_ln);
	else
		printf("Undefined source.");

	CG_init(u, d, r, p, bb, dx, dy, N, N_ln);

	// start evolution (selected method)
	double error = 1.0;
	//struct timespec start, end;
	clock_t start, end;
	//float time;
	//cudaEvent_t start, stop;

	printf("itr     error\n");
	printf("--------------\n");

	// prepare for GPU
	double *d_u, *d_p, *d_r, *d_Ap;
	// allocate device memory
	cudaMalloc(&d_u, (N * N) * sizeof(double));
	cudaMalloc(&d_p, (N * N) * sizeof(double));
	cudaMalloc(&d_r, (N_ln * N_ln) * sizeof(double));
	cudaMalloc(&d_Ap, (N_ln * N_ln) * sizeof(double));

	// transfer data from CPU to GPU
	cudaMemcpy(d_u, u, (N * N) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, p, (N * N) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, r, (N_ln * N_ln) * sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy( d_Ap, Ap, (N_ln*N_ln)*sizeof(double), cudaMemcpyHostToDevice );

	start = clock();
	while (error >= criteria)
	{
		if (method == 0)
			error = SOR(omega);
		else
			error = CG(d_u, d_p, d_r, d_Ap);

		itr++;
		if (itr >= 20000)
		{
			printf("Convergence Failure.\n");
			break;
		}
		if (itr % 100 == 0)
		{
			printf("%d      %1.3e\n", itr, error);
		}
	}
	end = clock();
	double time = double(end - start) / CLOCKS_PER_SEC;

	// transfer data from GPU to CPU
	cudaMemcpy(u, d_u, (N * N) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_u);
	cudaFree(d_p);
	cudaFree(d_r);
	cudaFree(d_Ap);

	// generate final result data for plotting
	WriteToFile(u, N);

	// print out result
	if (method == 0)
	{
		printf("\nSOR Poisson Solver\n");
		printf("----------------------------------\n");
		printf("opt omega = %f\n", omega);
	}
	else
	{
		printf("\nConjugate Gradient Poisson Solver\n");
		printf("----------------------------------\n");
	}
	printf("N = %d\n", N_ln);
	printf("GPU Parallel\n");
	printf("Iteration = %d, error = %1.3e\n", itr, error);
	printf("Iteration Wallclock time : %f s \n", time);

	delete[] u;
	delete[] d;
	delete[] r;
	delete[] p;
	delete[] Ap;

	return 0;
}

double SOR(double omega)
{
	double residual = 0.0;
	double dd = 0.0;

	{
		//	const int tid = omp_get_thread_num();
		//	const int nt  = omp_get_num_threads();
		// odd loop
		for (int i = 0; i < N_ln; i++)
		{
			double tmp_residual = 0.0;
			double tmp_d = 0.0;
			for (int j = i % 2; j < N_ln; j += 2)
			{
				double psy = (u[N * i + (j + 1)] + u[N * (i + 2) + (j + 1)] + u[N * (i + 1) + j] +
							  u[N * (i + 1) + (j + 2)] - 4.0 * u[N * (i + 1) + (j + 1)] - dx * dy * d[N_ln * i + j]);
				u[N * (i + 1) + (j + 1)] += 0.25 * omega * psy;
				//tmp_residual += fabs(psy/u[N*(i+1)+(j+1)])/(N_ln*N_ln);
				tmp_residual += fabs(dx * dy * psy);
				tmp_d += fabs(d[N_ln * i + j]);
			}
			residual += tmp_residual;
			dd += tmp_d;
			//			printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
		}

		// even loop
		for (int i = 0; i < N_ln; i++)
		{
			double tmp_residual = 0.0;
			double tmp_d = 0.0;
			for (int j = (i + 1) % 2; j < N_ln; j += 2)
			{
				double psy = (u[N * i + (j + 1)] + u[N * (i + 2) + (j + 1)] + u[N * (i + 1) + j] +
							  u[N * (i + 1) + (j + 2)] - 4.0 * u[N * (i + 1) + (j + 1)] - dx * dy * d[N_ln * i + j]);
				u[N * (i + 1) + (j + 1)] += 0.25 * omega * psy;
				//tmp_residual += fabs(psy/u[N*(i+1)+(j+1)])/(N_ln*N_ln);
				tmp_residual += fabs(dx * dy * psy);
				tmp_d += fabs(d[N_ln * i + j]);
			}
			residual += tmp_residual;
			dd += tmp_d;
			//			printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
		}
	}

	residual = residual / dd;

	return residual;
}

double CG(double *d_u, double *d_p, double *d_r, double *d_Ap)
{
	double pAp = 0.0;	// p*A*p
	double alpha = 0.0; // for update x (u)
	double beta = 0.0;	// for update pk (search direction)
	double rr0 = 0.0;	// old r*r
	double rr1 = 0.0;	// new r*r
	double err = 0.0;

	// execute the GPU kernel
	laplacian_GPU<<<dimGrid, dimBlock>>>(d_Ap, d_p, dx, dy, N, N_ln);

	// error handling
	cudaError_t ErrGPU = cudaGetLastError();
	if (ErrGPU != cudaSuccess)
	{
		printf("Kernel error: %s\n", cudaGetErrorString(ErrGPU));
		exit(EXIT_FAILURE);
	}

	// transfer data from GPU to CPU
	cudaMemcpy(Ap, d_Ap, (N_ln * N_ln) * sizeof(double), cudaMemcpyDeviceToHost);

	rr0 = inner_product(r,r,0,N,N_ln);		
	pAp = inner_product(p,Ap,1,N,N_ln);		// pAp	
	alpha = rr0 / pAp;

	// GPU start (rr0 & pAp)
	// execute the GPU kernel

	YPEAX_GPU<<<NBlock, NThread_Per_Block>>>(d_u, d_p, alpha, N);	   // update u
	YPEAX_GPU<<<NBlock, NThread_Per_Block>>>(d_r, d_Ap, -alpha, N_ln); // update r

	// error handling
	ErrGPU = cudaGetLastError();
	if (ErrGPU != cudaSuccess)
	{
		printf("Kernel error: %s\n", cudaGetErrorString(ErrGPU));
		exit(EXIT_FAILURE);
	}

	// transfer data from GPU to CPU
	//cudaMemcpy( u, d_u, (N*N)*sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy(r, d_r, (N_ln * N_ln) * sizeof(double), cudaMemcpyDeviceToHost);
	// GPU end (rr0 & pAp)

	rr1 = inner_product(r,r,0,N,N_ln);
	beta = rr1 / rr0;

	// GPU start (update p)
	// execute the GPU kernel
	YEAYPX_GPU<<<dimGrid, dimBlock>>>(d_p, d_r, beta, N, N_ln);
	// error handling
	ErrGPU = cudaGetLastError();
	if (ErrGPU != cudaSuccess)
	{
		printf("Kernel error: %s\n", cudaGetErrorString(ErrGPU));
		exit(EXIT_FAILURE);
	}

	// transfer data from GPU to CPU
	cudaMemcpy(p, d_p, (N * N) * sizeof(double), cudaMemcpyDeviceToHost);
	// GPU end (update p)

	rr0 = rr1;
	err = sqrt(rr1 / bb);

	return err;
}
