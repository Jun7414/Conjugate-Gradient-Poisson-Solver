double CG()
{
	double pAp = 0.0;	// p*A*p
	double alpha = 0.0;	// for update x (u)
	double beta = 0.0;	// for update pk (search direction)
	double rr0 = 0.0;	// old r*r
	double rr1 = 0.0;	// new r*r
	double err = 0.0;

	rr0 = inner_product(r, r, 0, N, N_ln);
	laplacian(Ap, p, dx, dy, N, N_ln);		// A.p
	pAp = inner_product(p, Ap, 1, N, N_ln); 	// pAp
	alpha = rr0 / pAp;

	#pragma omp parallel for
	for (int i = 0; i < N_ln; i++)
	{
		for (int j = 0; j < N_ln; j++)
		{
			u[N * (i + 1) + (j + 1)] += alpha * p[N * (i + 1) + (j + 1)]; 		// update u
			r[N_ln * i + j] += -alpha * Ap[N_ln * i + j];				// update r
		}
	}

	rr1 = inner_product(r, r, 0, N, N_ln);
	beta = rr1 / rr0;
	#pragma omp parallel for
	for (int i = 0; i < N_ln; i++)
	{
		for (int j = 0; j < N_ln; j++)
		{
			p[N * (i + 1) + (j + 1)] = r[N_ln * i + j] + beta * p[N * (i + 1) + (j + 1)];
		}
	}
	rr0 = rr1;
	//printf("err = %2.15f\n",rr1);

	err = sqrt(rr1 / bb);

	return err;
}
