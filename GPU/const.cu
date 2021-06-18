// User define constants
const int method  = 1;			// 0 = SOR; 1 = CG
const int N_ln    = 512;		// grid size
const int bc      = 0;			// 0: const_bc,  1: one_bc, 2:four_bc, 3:sin_bc
const int source   = 1;			// 0:background density, 1:point source middle, 2: point source 4th quadrant
const double criteria = 1.0e-14;        // convergence criteria
