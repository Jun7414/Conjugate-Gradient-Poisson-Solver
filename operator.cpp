
double inner_product(double *a, double *b, int type, int N, int N_ln)
{
    double kk = 0.0;
    int i, j;
    
    if (type == 0)
    { // for N_ln^2 * N_ln^2

        #pragma omp parallel for private(i) shared( a, b, N, N_ln ) reduction( +:kk )
        for ( i = 0; i < N_ln * N_ln; i++)
        {
            kk += a[i] * b[i];
        }

    }
    else
    { // for N^2 * N_ln^2

        #pragma omp parallel for private(i, j) shared( a, b, N, N_ln ) reduction( +:kk )
        for ( i = 0; i < N_ln; i++)
        {
            for ( j = 0; j < N_ln; j++)
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

    #pragma omp parallel for private(i, j) shared( La, x, dx, dy, N, N_ln )
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

void YPEAX(double *y, double *x, double a, int N) // Y += a*X
{
    int i;

    #pragma omp parallel for private(i) shared( y, x, a, N )
    for ( i = 0; i < N * N; i++)
    {
        y[i] += a * x[i];
    }

    return;
}

void YEAYPX(double *y, double *x, double a, int N, int N_ln) // Y = a*Y + X
{
    int i, j;
    
    #pragma omp parallel for private(i, j) shared( y, x, a, N, N_ln ) 
    for ( i = 0; i < N_ln; i++)
    {
        for ( j = 0; j < N_ln; j++)
        {
            y[N * (i + 1) + (j + 1)] = a * y[N * (i + 1) + (j + 1)] + x[N_ln * i + j];
        }
    }

    return;
}
