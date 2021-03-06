// User define constants

int Nthread = 8;				 // MPI parallel with N threads
int N_ln = 1024;				 // grid size
const double criteria = 1.0e-14; // convergence criteria
const int bc = 0;				 // 0: const_bc,  1: one_bc, 2:four_bc, 3:sin_bc
const int source = 2;			 // 0:background density, 1:point source middle, 2: point source 4th quadrant
//

// constants
const double Lx = 1.0;			// x computational domain size
const double Ly = 1.0;			// y computational domain size
const int ghost = 1;			// ghost zones
const int N = N_ln + 2 * ghost; // total number of cells including ghost zones
const double u0 = 1.0;			// background density
const double omega = 4.0 / (2.0 + sqrt(4.0 -
									   (2.0 * cos(M_PI / N_ln)) * (2.0 * cos(M_PI / N_ln))));
int itr = 0;	 // count iteration
double bb = 0.0; // criteria standard

// derived constants
const double dx = Lx / N_ln; // spatial resolution
const double dy = Ly / N_ln;

// pointer
double *u = new double[N * N];		  // potential
double *d = new double[N_ln * N_ln];  // density
double *r = new double[N_ln * N_ln];  // residual vector
double *p = new double[N * N];		  // search direction
double *Ap = new double[N_ln * N_ln]; // <A|p>

const struct option long_options[] =
{
		{"SOR", 1, NULL, 's'},
		{"CG", 1, NULL, 'c'},
		{"N", 1, NULL, 'n'},
		{0, 0, 0, 0},
};
