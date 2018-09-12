// This routine is from Numerical Recipes, see the .cpp file for notes.

#ifndef DOWNHILL_SIMPLEX_H
#define DOWNHILL_SIMPLEX_H

#define   MP  22
#define   NP  21    //Maximum value for NDIM=20
typedef   double MAT[MP][NP];

// We define this function.
double FUNC(double *P);

// This runs the downhill simplex routine.
void AMOEBA(MAT P, double *Y, int NDIM, double FTOL, int *ITER);

#endif

