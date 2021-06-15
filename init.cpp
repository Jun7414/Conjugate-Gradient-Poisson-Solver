
void CG_init(double *x, double *b, double *r, double *p, double &bb,
             double dx, double dy, int N, int N_ln)
{
    double *Ax = new double[N_ln * N_ln];

    laplacian(Ax, x, dx, dy, N, N_ln);
    bb = inner_product(b, b, 0, N, N_ln);
    bb += 1e-15; // avoid being 0 in rr/bb

#pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
        p[i] = 0.0;
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_ln; i++)
    {
        for (int j = 0; j < N_ln; j++)
        {
            r[N_ln * i + j] = b[N_ln * i + j] - Ax[N_ln * i + j];
            p[N * (i + 1) + (j + 1)] = r[N_ln * i + j];
        }
    }

    delete[] Ax;
    return;
}

void const_bc(double *u, double u0, int N)
{
#pragma omp parallel for
    
    
    for (int i = 0; i < N; i++) // Work
    {
        for( int j = 0; j < N; j++ )
        {
            if( i < 10 || i > N - 10 || j < 10 || j > N-10 )
            {
                u[i * N + j] = sin(M_PI * i * j);
            }
            else
            {
                u[i * N + j] = 1.0;   
            }
           
        }
    }

    /*
    for (int i = 0; i < N; i++) // work
    {
        for (int j = 0; j < N; j++)
        {

            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
            {
                u[i * N + j] = 1.0;
            }
            else
            {
                u[i * N + j] = 0.0;
            }
        }
    }
    */

    
    /*for (int i = 0; i < N * N; i++) // Work
    {
        u[i] = u0;
    }*/
    
    return;
}

void point_source(double *d, int N_ln)
{
    const double G = 1.0;

#pragma omp parallel for collapse(2)


    for (int i = N_ln / 2 - 1; i <= N_ln / 2; i++)
    {
        for (int j = N_ln / 2 - 1; j <= N_ln / 2; j++)
        {
            d[N_ln * i + j] = 4 * M_PI * G * 5.0;
       
        }
    }

    /*for( int i = 1; i < N_ln; i++ )
    {
        for( int j = 0; j < N_ln; j++ )
        {
            d[i * N_ln + j] = -( M_PI * M_PI * ( i * i + j * j ) * sin( M_PI * i * j ));
        }
    }*/

    return;
}
