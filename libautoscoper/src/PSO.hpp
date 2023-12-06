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
#include <ostream>
#include <vector>
#include "Particle.hpp"
const int NUM_OF_PARTICLES = 100;
const int NUM_OF_DIMENSIONS = 6;
// END: NEW PSO

double PSO_FUNC(double *P);

void intializeRandom();

float host_fitness_function(const std::vector<float>& x);

// Stream operator
extern std::ostream& operator<<(std::ostream& os, const std::vector<float>& values);

Particle* pso(float start_range_min, float start_range_max, unsigned int MAX_EPOCHS, unsigned int MAX_STALL);
