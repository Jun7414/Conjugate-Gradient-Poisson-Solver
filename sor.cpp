double SOR(double omega)
{
	double residual = 0.0;
	double dd = 0.0;

// odd loop
#pragma omp parallel for
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

#pragma omp critical
		residual += tmp_residual;
		dd += tmp_d;
		//printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
	}

// even loop
#pragma omp parallel for
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

#pragma omp critical
		residual += tmp_residual;
		dd += tmp_d;
		//printf( "loop %d is computed by thread %d/%d\n", itr, tid, nt );
	}

	residual = residual / dd;

	return residual;
}