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

// PSO Acceleration coefficients, generally should sum to 2.0
//const float c1 = 1.5f; // Attraction towards this particles own best position
//const float c2 = 1.5f; // Attraction towards this best position of all particles
const float c1 = 0.8f; // Attraction towards this particles own best position
const float c2 = 1.2f; // Attraction towards this best position of all particles
// END: NEW PSO


///
/// \brief Function that calls minimization function
/// 
double PSO_FUNC(double *P);

///
/// \brief Initialize seed for PSO optimization
/// 
void intializeRandom();

///
/// \brief Generate a random number between low and high
/// 
float getRandom(float low, float high);


///
/// \brief Generate a random number between 0 and 1
/// 
float getRandomClamped();


///
/// \brief Minimization function wrapper
/// \param x[] particle pose
/// 
float host_fitness_function(float x[]);


///
/// \brief Perform particle swarm optimization
/// \param positions Array size [NUM_OF_PARTICLES*NUM_OF_DIMENSIONS] of particle poses (positions)
/// \param velocities Array size [NUM_OF_PARTICLES*NUM_OF_DIMENSIONS] of particle velocities
/// \param pBest array Size [NUM_OF_PARTICLES*NUM_OF_DIMENSIONS] of best pose for each particle
/// \param gBest array Size [NUM_OF_DIMENSIONS] of the global best pose
/// \param MAX_EPOCHS Max number of iterations before exiting pso
/// \param MAX_STALL Max number of iterations where the NCC value changes by a small value (1E-4)
///
void pso(float *positions, float *velocities, float *pBests, float *gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL);
