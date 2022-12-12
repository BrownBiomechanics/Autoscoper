/*
 * pso_kernel.h
 *
 *  Created on: Jul 27, 2017
 *      Author: root
 */

#pragma once

 // ADD: NEW PSO
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
const int NUM_OF_PARTICLES = 100;
const int NUM_OF_DIMENSIONS = 6;
const float c1 = 1.5f;
const float c2 = 1.5f;
// END: NEW PSO

double PSO_FUNC(double *P);

void intializeRandom();

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);

void pso(float *positions, float *velocities, float *pBests, float *gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL);
