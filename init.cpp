
void CG_init(double *x, double *b, double *r, double *p, double &bb,
             double dx, double dy, int N, int N_ln)
{
    double *Ax = new double[N_ln * N_ln];
    int i, j;

    laplacian(Ax, x, dx, dy, N, N_ln);
    bb = inner_product(b, b, 0, N, N_ln);
    bb += 1e-15; // avoid being 0 in rr/bb

#pragma omp parallel for private(i) shared(p, N)
    for (i = 0; i < N * N; i++)
    {
        p[i] = 0.0;
    }

#pragma omp parallel for private(i, j) shared(Ax, b, r, p, N_ln)
    for (i = 0; i < N_ln; i++)
    {
        for (j = 0; j < N_ln; j++)
        {
            r[N_ln * i + j] = b[N_ln * i + j] - Ax[N_ln * i + j];
            p[N * (i + 1) + (j + 1)] = r[N_ln * i + j];
        }
    }

    delete[] Ax;
    return;
}

void oneside_bc(double *u, double u0, int N)
{
    int i, j;
#pragma omp parallel for private(i, j) shared(N, u, u0)
    for (i = 0; i < N; i++) // work
    {
        for (j = 0; j < N; j++)
        {

            if (i == 0)
            {
                u[i * N + j] = 5.0;
            }
            else
            {
                u[i * N + j] = u0;
            }
        }
    }
    return;
}

void fourside_bc(double *u, double u0, int N)
{
    int i, j;
#pragma omp parallel for private(i, j) shared(N, u, u0)
    for (i = 0; i < N; i++) // work
    {
        for (j = 0; j < N; j++)
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
    return;
}

void sin_bc(double *u, double u0, int N)
{
#pragma omp parallel for
    for (int i = 0; i < N; i++) // Work
    {
        for (int j = 0; j < N; j++)
        {
            if (i == 0)
            {
                double theta = double(j) / double(N);
                u[i * N + j] = u0 + 5.0 * sin(2 * M_PI * theta);
            }
            else if (i == N - 1)
            {
                double theta = double(j) / double(N);
                u[i * N + j] = u0 + 5.0 * cos(2 * M_PI * theta);
            }
            else
            {
                u[i * N + j] = u0;
            }
        }
    }
    return;
}

void const_bc(double *u, double u0, int N)
{
    int i, j;
#pragma omp parallel for private(i) shared(N, u, u0)
    for (i = 0; i < N * N; i++)
    {
        u[i] = u0;
    }

    return;
}

void background_density(double *d, int N_ln)
{
    int i, j;
#pragma omp parallel for private(i, j) shared(d, N_ln)
    for (i = 0; i < N_ln; i++)
    {
        for (j = 0; j < N_ln; j++)
        {
            d[N_ln * i + j] = 1.0;
        }
    }
    return;
}

void point_source_4q(double *d, int N_ln)
{
    const double G = 1.0;
    int i, j;

#pragma omp parallel for private(i, j) shared(d, N_ln)
    for (i = N_ln / 4 - 1; i <= N_ln / 4; i++)
    {
        for (j = 3 * N_ln / 4 - 1; j <= 3 * N_ln / 4; j++)
        {
            d[N_ln * i + j] = 4 * M_PI * G * 2e5;
        }
    }
    return;
}

void point_source_middle(double *d, int N_ln)
{
    const double G = 1.0;
    int i, j;

#pragma omp parallel for private(i, j) shared(d, N_ln)
    for (i = N_ln / 2 - 1; i <= N_ln / 2; i++)
    {
        for (j = N_ln / 2 - 1; j <= N_ln / 2; j++)
        {
            d[N_ln * i + j] = 4 * M_PI * G * 2e5;
        }
    }
    return;
}
