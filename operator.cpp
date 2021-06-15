
double inner_product(double *a, double *b, int type, int N, int N_ln)
{
    double kk = 0.0;
    if (type == 0)
    { // for N_ln^2 * N_ln^2
#pragma omp parallel for reduction(+ : kk)
        for (int i = 0; i < N_ln * N_ln; i++)
        {
            kk += a[i] * b[i];
        }
    }
    else
    { // for N^2 * N_ln^2
#pragma omp parallel for reduction(+ : kk)
        for (int i = 0; i < N_ln; i++)
        {
            for (int j = 0; j < N_ln; j++)
            {
                kk += a[N * (i + 1) + (j + 1)] * b[N_ln * i + j];
            }
        }
    }
    return kk;
}

void laplacian(double *La, double *x, double dx, double dy, int N, int N_ln)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_ln; i++)
    {
        for (int j = 0; j < N_ln; j++)
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
#pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
        y[i] += a * x[i];
    }

    return;
}

void YEAYPX(double *y, double *x, double a, int N, int N_ln) // Y = a*Y + X
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_ln; i++)
    {
        for (int j = 0; j < N_ln; j++)
        {
            y[N * (i + 1) + (j + 1)] = a * y[N * (i + 1) + (j + 1)] + x[N_ln * i + j];
        }
    }

    return;
}
