// Purpose:
//      2D poisson solver with interative method ( Jacobi, Gauss-Seidel, SOR )
// 
// Version:
//      Testing
//
// Author :
//      2021 May 21, Nai Chieh Lin
//
// Compile:
//      gcc -fopenmp sor.c -o sor

// TODO : There are still some bugs to fix
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include "sor.h"

int main(int argc, char *argv[])
{
    bool converged;

    double wtime;
    double diff;
    double dx, dy;
    double error;
    double f[NX][NY];

    double u[NX][NY];
    double u_new[NX][NY];
    double u_exact[NX][NY];
    double u_diff[NX][NY];
    double u_norm;
    double u_new_norm;

    int i, j, x, y;
    int nx = NX, ny = NY;

    int old_iter, new_iter;

    printf("\n");
    printf("=== OPENMP Information ===\n");
    printf("The number of processors is %d\n", omp_get_num_procs());

    #pragma omp parallel
    {
        int id = omp_get_thread_num();

        if (id == 0)
        {
            printf("The maximum number of threads is %d\n", omp_get_num_threads());
        }
    }

    dx = 1.0 / (double)(nx - 1);
    dy = 1.0 / (double)(ny - 1);

    printInformations( nx, ny, dx, dy );

    rhs(nx, ny, f);

    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
            {
                u_new[i][j] = f[i][j];
            }
            else
            {
                u_new[i][j] = 0.0;
            }
        }
    }

    u_new_norm = norm(nx, ny, u_new);

    for (j = 0; j < ny; j++)
    {
        y = (double)(j) / (double)(ny - 1);

        for (i = 0; i < nx; i++)
        {
            x = (double)(i) / (double)(nx - 1);
            u_exact[i][j] = uexact(x, y);
        }
    }

    u_norm = norm(nx, ny, u_exact);
    printf(" norm of exact solution = %g\n", u_norm);

    converged = false;

    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            u_diff[i][j] = u_new[i][j] - u_exact[i][j];
        }
    }

    error = norm(nx, ny, u_diff);
    printf(" error = %g\n", error);

    wtime = omp_get_wtime();

    new_iter = 0;
    while (true)
    {
        old_iter = new_iter;
        new_iter = old_iter + 100;
        //jacobi( nx, ny, dx, dy, f, old_iter, new_iter, u, u_new );
        //gauess_seidel( nx, ny, dx, dy, f, old_iter, new_iter, u, u_new );
        //sor_v1( nx, ny, dx, dy, f, old_iter, new_iter, u, u_new );
        sor_v2( nx, ny, dx, dy, f, old_iter, new_iter, u, u_new );

        u_norm = u_new_norm;
        u_new_norm = norm( nx, ny, u_new );

        for (j = 0; j < ny; j++)
        {
            for (i = 0; i < nx; i++)
            {
                u_diff[i][j] = u_new[i][j] - u[i][j];
            }
        }

        diff = norm(nx, ny, u_diff);

        for (j = 0; j < ny; j++)
        {
            for (i = 0; i < nx; i++)
            {
                u_diff[i][j] = u_new[i][j] - u_exact[i][j];
            }
        }

        error = norm(nx, ny, u_diff);

        printf("%d\t%g\t%g\t%g\n", new_iter, u_new_norm, diff, error);
        
        if (diff <= TOLERANCE)
        {
            converged = true;
            break;
        }
    }

    if( converged )
    {
        printf("The iteration has converged\n");
    }
    else
    {
        printf("The iteration has not converged\n");
    }

    wtime = omp_get_wtime() - wtime;
    printf("Elapsed Time = %g sec\n", wtime);
    return 0;
}

void jacobi( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] )
{
    int i, j, iter;

    #pragma omp parallel \
    shared( dx, dy, new_iter, old_iter, nx, ny, u, u_new ) \
    private( i, j, iter )

    for( iter = old_iter + 1; iter <= new_iter; iter ++ )
    {
        #pragma omp for
        for ( j = 0; j < ny; j++ )
        {
            for( i = 0 ; i < nx; i++ )
            {
                u[i][j] = u_new[i][j];
            }
        }

        #pragma omp for
        for( j = 0; j < ny; j++ )
        {
            for( i = 0; i < nx; i++ )
            {
                if( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
                {
                    u_new[i][j] = f[i][j];
                }
                else
                {
                    u_new[i][j] =  0.25 * ( u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy );
                }
            }
        }
    }
    
    return;
}

void gauess_seidel( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] )
{
    int i, j, iter;
    
    #pragma omp parallel \
    shared( dx, dy, new_iter, old_iter, nx, ny, u, u_new ) \
    private( i, j, iter )

    for( iter = old_iter + 1; iter <= new_iter; iter ++ )
    {
        #pragma omp for
        for ( j = 0; j < ny; j++ )
        {
            for( i = 0 ; i < nx; i++ )
            {
                u[i][j] = u_new[i][j];
            }
        }

        #pragma omp for
        for( j = 0; j < ny; j++ )
        {
            for( i = 0; i < nx; i++ )
            {
                if( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
                {
                    u_new[i][j] = f[i][j];
                }
                else
                {
                    u_new[i][j] =  0.25 * ( u_new[i-1][j] + u[i][j+1] + u_new[i][j-1] + u[i+1][j] + f[i][j] * dx * dy );
                }
            }
        }
    }

    return;
}

void sor_v1( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] )
{
    int i, j, iter;
    
    #pragma omp parallel \
    shared( dx, dy, new_iter, old_iter, nx, ny, u, u_new ) \
    private( i, j, iter )

    for( iter = old_iter + 1; iter <= new_iter; iter ++ )
    {
        #pragma omp for
        for ( j = 0; j < ny; j++ )
        {
            for( i = 0 ; i < nx; i++ )
            {
                u[i][j] = u_new[i][j];
            }
        }

        #pragma omp for
        for( j = 0; j < ny; j++ )
        {
            for( i = 0; i < nx; i++ )
            {
                if( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 ) // Boundary Condition
                {
                    u_new[i][j] = f[i][j];
                }
                else {
                    if( ( i + j ) % 2 == 0 ) // Black
                    {
                        u_new[i][j] =  0.25 * ( u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy );
                    }

                    if( ( i + j ) % 2 != 0 ) // Red
                    {
                        u_new[i][j] =  0.25 * ( u_new[i-1][j] + u_new[i][j+1] + u_new[i][j-1] + u_new[i+1][j] + f[i][j] * dx * dy );
                    }
                }
            }
        }
    }
    
    return;
}

void sor_v2( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] )
{
    int i, j, iter;
    
    #pragma omp parallel \
    shared( dx, dy, new_iter, old_iter, nx, ny, u, u_new ) \
    private( i, j, iter )

    for( iter = old_iter + 1; iter <= new_iter; iter ++ )
    {

        #pragma omp for
        for ( j = 0; j < ny; j++ )
        {
            for( i = 0 ; i < nx; i++ )
            {
                u[i][j] = u_new[i][j];
            }
        }

        #pragma omp for
        for( j = 0; j < ny; j++ )
        {
            for( i = 0; i < nx; i++ )
            {
                if( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )  // Boundary Condition
                {
                    u_new[i][j] = f[i][j];
                }
                else
                {
                    if( ( i + j ) % 2 == 0 ) // Black
                    {
                        u_new[i][j] =  u[i][j] + w * 0.25 * ( u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy - 4 * u[i][j] );
                    }

                    if( ( i + j ) % 2 != 0 ) // Red
                    {
                        u_new[i][j] =  u[i][j] + w * 0.25 * ( u_new[i-1][j] + u_new[i][j+1] + u_new[i][j-1] + u_new[i+1][j] + f[i][j] * dx * dy - 4 * u[i][j] );
                    }
                }
            }
        }
    }
    
    return;
}

void rhs(int nx, int ny, double f[NX][NY])
{
    int i, j;
    double x, y;

    for (j = 0; j < ny; j++)
    {
        y = (double)(i) / (double)(ny - 1);

        for (i = 0; i < nx; i++)
        {
            x = (double)(i) / (double)(nx - 1);
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
            {
                f[i][j] = uexact(x, y);
            }
            else
            {
                f[i][j] = -uxxyy_exact(x, y);
            }
        }
    }

    double fnorm = norm(nx, ny, f);
    printf(" norm of F = %g\n", fnorm);
    return;
}

double uexact(double x, double y)
{
    return sin(pi * x * y);
}

double uxxyy_exact(double x, double y)
{
    return -(pi * pi * (x * x + y * y) * sin(pi * x * y));
}

double norm(int nx, int ny, double array[NX][NY])
{
    int i, j;
    double sum = 0.0;

    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            sum = sum + array[i][j] * array[i][j];
        }
    }

    sum = sqrt(sum / (double)(nx * ny));
    return sum;
}

void printInformations( int nx, int ny, double dx, double dy )
{
    printf("\n");
    printf("=== Equation Information ===\n");
    printf("-DEL^2 U = F(X,Y)\n");
    printf("0 <= X <= 1, 0 <= Y <= 1\n");
    printf("F(X,Y) = pi^2 * ( x^2 + y^2 ) * sin( pi *x * y )\n");
    printf("\n");
    printf("=== Grid Information ===\n");
    printf("The number of interior X grid points is %d\n", nx);
    printf("The number of interior Y grid points is %d\n", ny);
    printf("The X grid spacing is %f\n", dx);
    printf("The Y grid spacing is %f\n", dy);
}