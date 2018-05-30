// dfk: Grabbed this code form the webpage "Functional Approximations
// in C/C++" http://perso.orange.fr/jean-pierre.moreau/c_function.html
// which appears to have derived it from Numerical Recipes.
#include "DownhillSimplex.hpp"

/**********************************************************
*   Multidimensional minimization of a function FUNC(X)   *
*  where X is an NDIM-dimensional vector, by the downhill *
*  simplex method of Nelder and Mead.                     *
* ------------------------------------------------------- *
* SAMPLE RUN: Find a minimum of function F(x,y):          *
*             F=Sin(R)/R, where R = Sqrt(x*x+y*y).        *
*                                                         *
*  Number of iterations: 22                               *
*                                                         *
*  Best NDIM+1 points:                                    *
*   4.122686  1.786915                                    *
*   4.166477  1.682698                                    *
*   4.142454  1.741176                                    *
*                                                         *
*  Best NDIM+1 mimimum values:                            *
*   -0.2172336265                                         *
*   -0.2172336281                                         *
*   -0.2172336271                                         *
*                                                         *
* ------------------------------------------------------- *
* REFERENCE: "Numerical Recipes, The Art of Scientific    *
*             Computing By W.H. Press, B.P. Flannery,     *
*             S.A. Teukolsky and W.T. Vetterling,         *
*             Cambridge University Press, 1986"           *
*             [BIBLI 08].                                 *
*                                                         *
*                      C++ Release By J-P Moreau, Paris.  *
**********************************************************/
#include <stdio.h>
#include <cmath>
#include <iostream>

void AMOEBA(MAT P, double *Y, int NDIM, double FTOL, int *ITER, double ALPHA, double GAMMA, double BETA) {
/*-------------------------------------------------------------------
! Multidimensional minimization of the function FUNC(X) where X is
! an NDIM-dimensional vector, by the downhill simplex method of
! Nelder and Mead. Input is a matrix P whose NDIM+1 rows are NDIM-
! dimensional vectors which are the vertices of the starting simplex
! (Logical dimensions of P are P(NDIM+1,NDIM); physical dimensions
! are input as P(NP,NP)). Also input is the vector Y of length NDIM
! +1, whose components must be pre-initialized to the values of FUNC
! evaluated at the NDIM+1 vertices (rows) of P; and FTOL the fractio-
! nal convergence tolerance to be achieved in the function value. On
! output, P and Y will have been reset to NDIM+1 new points all within
! FTOL of a minimum function value, and ITER gives the number of ite-
! rations taken.
!-------------------------------------------------------------------*/
// Label:  e1
const int NMAX=20, ITMAX=5000;//ITMAX=500;
//Expected maximum number of dimensions, three parameters which define
//the expansions and contractions, and maximum allowed number of
//iterations.
  double PR[MP], PRR[MP], PBAR[MP];
  //double ALPHA=1.0, BETA=0.5, GAMMA=2.0;
  int I,IHI,ILO,INHI,J,MPTS;
  double RTOL,YPR,YPRR;
  MPTS=NDIM+1;
  *ITER=0;
e1:ILO=1;
  if (Y[1] > Y[2]) {
    IHI=1;
    INHI=2;
  }
  else {
    IHI=2;
    INHI=1;
  }
  for (I=1; I<=MPTS; I++) {
    if (Y[I] < Y[ILO])  ILO=I;
    if (Y[I] > Y[IHI]) {
      INHI=IHI;
      IHI=I;
    }
    else if (Y[I] > Y[INHI])
      if (I != IHI)  INHI=I;
  }
  //Compute the fractional range from highest to lowest and return if
  //satisfactory.
  //RTOL=2.0*fabs(Y[IHI]-Y[ILO])/(fabs(Y[IHI])+fabs(Y[ILO]));

  // Use the average distance from the centroid as the stopping criteria
  // Calculate the centroid of the simplex

  double CENT[NP];
  for (J=1; J<=NDIM; J++) {
    CENT[J] = 0;
    for (I=1; I<=MPTS; I++)
      CENT[J] += P[I][J];
    CENT[J] /= MPTS;
  }

  RTOL = 0;
  for (I=1; I<=MPTS; I++) {
    double ss = 0;
    for (J=1; J<=NDIM; J++) {
        ss += (CENT[J]-P[I][J])*(CENT[J]-P[I][J]);
    }
    RTOL += sqrt(ss);
  }
  RTOL /= MPTS;

  if (RTOL < FTOL)  {
	  printf("The final pose tolerance is: %.3e\n", RTOL);
	  return;  //normal exit
  }
  if (*ITER == ITMAX) {
    printf("Optimization exceeding maximum iterations. Pose tolerance is: %.3e\n",RTOL);
    return;
  }
  *ITER= (*ITER) + 1;
  for (J=1; J<=NDIM; J++)  PBAR[J]=0.0;
  for (I=1; I<=MPTS; I++)
    if (I != IHI)
      for (J=1; J<=NDIM; J++)
        PBAR[J] += P[I][J];
  for (J=1; J<=NDIM; J++) {
    PBAR[J] /= 1.0*NDIM;
    PR[J]=(1.0+ALPHA)*PBAR[J] - ALPHA*P[IHI][J];
  }
  YPR=FUNC(PR);
  if (YPR <= Y[ILO]) {
    for (J=1; J<=NDIM; J++)
      PRR[J]=GAMMA*PR[J] + (1.0-GAMMA)*PBAR[J];
    YPRR=FUNC(PRR);
    if (YPRR < Y[ILO]) {
      for (J=1; J<=NDIM; J++) P[IHI][J]=PRR[J];
      Y[IHI]=YPRR;
    }
    else {
      for (J=1; J<=NDIM; J++) P[IHI][J]=PR[J];
      Y[IHI]=YPR;
	}
  }
  else if (YPR >= Y[INHI]) {
	if (YPR < Y[IHI]) {
      for (J=1; J<=NDIM; J++)  P[IHI][J]=PR[J];
      Y[IHI]=YPR;
    }
    for (J=1; J<=NDIM; J++)  PRR[J]=BETA*P[IHI][J] + (1.0-BETA)*PBAR[J];
    YPRR=FUNC(PRR);
    if (YPRR < Y[IHI]) {
      for (J=1; J<=NDIM; J++)  P[IHI][J]=PRR[J];
      Y[IHI]=YPRR;
    }
    else
      for (I=1; I<=MPTS; I++)
		if (I != ILO) {
		  for (J=1; J<=NDIM; J++) {
            PR[J]=0.5*(P[I][J] + P[ILO][J]);
	        P[I][J]=PR[J];
		  }
          Y[I]=FUNC(PR);
		}
  }
  else {
    for (J=1; J<=NDIM; J++)  P[IHI][J]=PR[J];
    Y[IHI]=YPR;
  }
  goto e1;
}
