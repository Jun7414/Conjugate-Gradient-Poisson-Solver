#define NX 321
#define NY 321
#define TOLERANCE 0.000001
#define pi 3.1415926
#define w 1.33

void jacobi( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] );
void gauess_seidel( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] );
void sor_v1( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] );
void sor_v2( int nx, int ny, double dx, double dy, double f[NX][NY], int old_iter, int new_iter, double u[NX][NY], double u_new[NX][NY] );
void rhs(int nx, int ny, double f[NX][NY]);
double uexact(double x, double y);
double uxxyy_exact(double x, double y);
double norm(int nx, int ny, double array[NX][NY]);
void printInformations( int nx, int ny, double dx, double dy );