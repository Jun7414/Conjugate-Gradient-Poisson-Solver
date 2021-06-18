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
	
	YPEAX(u,p,alpha,N);		// update u 
	YPEAX(r,Ap,-alpha,N_ln);	// update r
	
	rr1 = inner_product(r, r, 0, N, N_ln);
	beta = rr1 / rr0;
	YEAYPX(p,r,beta,N,N_ln);	// update p
	
	rr0 = rr1;
	err = sqrt(rr1 / bb);
	return err;
}
