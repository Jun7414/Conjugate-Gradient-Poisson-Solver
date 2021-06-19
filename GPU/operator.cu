/*#include "device_atomic_functions.h"
__global__
void inner_product1_GPU(double *kk, double *a, double *b, int N, int N_ln)
{
    *kk = 0.0;
    double tmpSum = 0.0;

    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if( row < N_ln && col < N_ln )
    {
        for( int i = 0; i < N_ln; i++)
        {
            tmpSum += a[ row  * N_ln + i] * b[ i * N_ln + col ];
        }
        *kk += tmpSum;
    }
   
    // for ( i = 0; i < N_ln*N_ln; i++)
    // if (i < N_ln*N_ln)
    // temp[threadIdx.x] = a[i] * b[i];

    // __syncthreads();
    // parallel reduction
    // for (int j = threadIdx.x; j < N_ln*N_ln ; j += blockDim.x)

    return;
}

__global__
void inner_product2_GPU(double *kk, double *a, double *b, int N, int N_ln)
{
    *kk = 0.0;
    double tmpSum = 0.0;

    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    const int col = blockIdx.y*blockDim.y + threadIdx.y;

    if( row < N_ln && col < N_ln )
    {
        for( int i = 0; i < N_ln; i++)
        {
            tmpSum += a[ (row + 1) * N + ( i + 1 ) ] * b[ i * N_ln + col ];
        }
        *kk += tmpSum;
    }

    // for ( i = 0; i < N_ln; i++)
    // for ( j = 0; j < N_ln; j++)
    // if (i < N_ln && j < N_ln)
    // kk += a[N * (i + 1) + (j + 1)] * b[N_ln * i + j] 
   
    return;
}*/

__global__
void laplacian_GPU(double *La, double *x, double dx, double dy, int N, int N_ln)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i < N_ln && j < N_ln)  
    La[N_ln * i + j] = (x[N * i + (j + 1)] + x[N * (i + 2) + (j + 1)] + x[N * (i + 1) + j] +
                        x[N * (i + 1) + (j + 2)] - 4.0 * x[N * (i + 1) + (j + 1)]) / (dx * dy);
  
    return;
}

__global__
void YPEAX_GPU(double *y, double *x, double a, int N) // Y += a*X
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    //for ( i = 0; i < N * N; i++)
    if (i< N*N)
    y[i] += a * x[i];

    return;
}

__global__
void YEAYPX_GPU(double *y, double *x, double a, int N, int N_ln) // Y = a*Y + X
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;    
    //for ( i = 0; i < N_ln; i++)
    //for ( j = 0; j < N_ln; j++)
    if (i < N_ln && j < N_ln)
    y[N * (i + 1) + (j + 1)] = a * y[N * (i + 1) + (j + 1)] + x[N_ln * i + j];

    return;
}

double inner_product(double *a, double *b, int type, int N, int N_ln)
{
    double kk = 0.0;

    if (type == 0)
    { // for N_ln^2 * N_ln^2
        for ( int i = 0; i < N_ln * N_ln; i++)
        {
            kk += a[i] * b[i];
        }
    }
    else
    { // for N^2 * N_ln^2
        for ( int i = 0; i < N_ln; i++)
        {
            for ( int j = 0; j < N_ln; j++)
            {
                kk += a[N * (i + 1) + (j + 1)] * b[N_ln * i + j];
            }
        }
    }
    return kk;
}

void laplacian(double *La, double *x, double dx, double dy, int N, int N_ln)
{
    int i, j;

    for ( i = 0; i < N_ln; i++)
    {
        for ( j = 0; j < N_ln; j++)
        {
            La[N_ln * i + j] = (x[N * i + (j + 1)] + x[N * (i + 2) + (j + 1)] + x[N * (i + 1) + j] +
                                x[N * (i + 1) + (j + 2)] - 4.0 * x[N * (i + 1) + (j + 1)]) /
                               (dx * dy);
        }
    }
    return;
}

void YEAYPX(double *y, double *x, double a, int N, int N_ln) // Y = a*Y + X
{
 //   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   // const int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i,j;
    for ( i = 0; i < N_ln; i++)
    for ( j = 0; j < N_ln; j++)
    {
        y[N * (i + 1) + (j + 1)] = a * y[N * (i + 1) + (j + 1)] + x[N_ln * i + j];
    }
    return;
}
