// User define constants
const int method  = 1;			// 0 = SOR; 1 = CG
const int Nthread = 8;                  // MPI parallel with N threads
const int N_ln    = 1024;		// grid size
const int bc      = 1;			// 0: const_bc,  1: one_bc, 2:four_bc, 3:sin_bc
const int source   = 0;			// 0:background density, 1:point source middle, 2: point source 4th quadrant
const double criteria = 1.0e-14;        // convergence criteria
