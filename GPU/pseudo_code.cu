main()
{
	while (error >= criteria){
               
                error = CG();

                itr ++;
        }
}

CG()
{
// GPU start (A.p)
       *double *d_u, *d_p, *d_r, *d_Ap;

     // allocate device memory
        cudaMalloc( &d_u, (N*N)*sizeof(double) );

     // transfer data from CPU to GPU
        cudaMemcpy( d_u, u, (N*N)*sizeof(double), cudaMemcpyHostToDevice );

     // execute the GPU kernel
        operator_GPU <<< dimGrid, dimBlock >>> (d_Ap, d_p, dx, dy, N, N_ln);

     // transfer data from GPU to CPU
        cudaMemcpy( Ap, d_Ap, (N_ln*N_ln)*sizeof(double), cudaMemcpyDeviceToHost );

	cudafree(d_u);
}



////////////////////////////////////////////////


main()
{
	// allocate device memory
        cudaMalloc( &d_u, (N*N)*sizeof(double) );

     	// transfer data from CPU to GPU
        cudaMemcpy( d_u, u, (N*N)*sizeof(double), cudaMemcpyHostToDevice );
        
	while (error >= criteria){

                error = CG();

                itr ++;
        }

	// transfer data from GPU to CPU
        cudaMemcpy( u, d_u, (N*N)*sizeof(double), cudaMemcpyDeviceToHost );
	cudafree(d_u);
}

CG()
{ 
	// execute the GPU kernel
        operator_GPU <<< dimGrid, dimBlock >>> (d_Ap, d_p, dx, dy, N, N_ln);

	// transfer data from GPU to CPU
        cudaMemcpy( Ap, d_Ap, (N_ln*N_ln)*sizeof(double), cudaMemcpyDeviceToHost );

	// CPU calculation
	rr0 = inner_product(r,r,0,N,N_ln);
        pAp = inner_product(p,Ap,1,N,N_ln);      // pAp
        alpha = rr0/pAp;
}


















