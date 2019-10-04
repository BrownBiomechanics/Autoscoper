/*
 * pso_kernel.h
 *
 *  Created on: Jul 27, 2017
 *      Author: root
 */

#ifndef PSO_KERNEL_H
#define PSO_KERNEL_H

 // ADD: NEW PSO
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
const int NUM_OF_PARTICLES = 120;
const int NUM_OF_DIMENSIONS = 6;
const float OMEGA = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;
// END: NEW PSO

double PSO_FUNC(double *P);

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);

void pso(float *positions, float *velocities, float *pBests, float *gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL);


extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL);



#endif /* PSO_KERNEL_H */
