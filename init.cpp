
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
    
/*    
    for (int i = 0; i < N; i++) // Work
    {
        for( int j = 0; j < N; j++ )
        {
            if( i == 0 )
            {
		double theta = double(j)/double(N);
                u[i * N + j] = u0 + 5.0 * sin(2*M_PI * theta);
            }
	    else if (i == N-1)
	    {
		double theta = double(j)/double(N);
		u[i * N + j] = u0 + 5.0 * cos(2*M_PI * theta);
            }
            else
            {
                u[i * N + j] = u0;   
            }
           
        }
    }
*/
    
/*    // 4 side same 
    for (int i = 0; i < N; i++) // work
    {
        for (int j = 0; j < N; j++)
        {

            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
            {
                u[i * N + j] = 5.0;
            }
            else
            {
                u[i * N + j] = u0;
            }
        }
    }
*/
/*   
    // 3 side same 1 side different
    for (int i = 0; i < N; i++) // work
    {
        for (int j = 0; j < N; j++)
        {

            if (i == 0 )
            {
                u[i * N + j] = 5.0;
            }
            else
            {
                u[i * N + j] = u0;
            }
        }
    }
*/
  
    for (int i = 0; i < N * N; i++) // Work
    {
        u[i] = u0;
    }

    return;
}

void point_source(double *d, int N_ln)
{
    const double G = 1.0;

#pragma omp parallel for collapse(2)

    // point source 1 (middle)
    for (int i = N_ln / 2 - 1; i <= N_ln / 2; i++)
    {
        for (int j = N_ln / 2 - 1; j <= N_ln / 2; j++)
        {
            d[N_ln * i + j] = 4 * M_PI * G * 2e5;
       
        }
    }

/*    // point source 2 (4th quadrant)
    for (int i = N_ln / 4 - 1; i <= N_ln / 4; i++)
    {
        for (int j = 3*N_ln / 4 - 1; j <= 3*N_ln / 4; j++)
        {
            d[N_ln * i + j] = 4 * M_PI * G * 2e5;

        }
    }
*/
    /*for( int i = 1; i < N_ln; i++ )
    {
        for( int j = 0; j < N_ln; j++ )
        {
            d[i * N_ln + j] = -( M_PI * M_PI * ( i * i + j * j ) * sin( M_PI * i * j ));
        }
    }*/
/*
    // back ground density only  (for bc test)
    for (int i = 0; i < N_ln; i++)
    {
        for (int j = 0; j < N_ln; j++)
	{
		d[N_ln * i + j] = 1.0;
	}
    }
*/
    return;
}
