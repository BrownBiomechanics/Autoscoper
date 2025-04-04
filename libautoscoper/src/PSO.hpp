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
const int NUM_OF_PARTICLES = 100;
const int NUM_OF_DIMENSIONS = 6;
const float c1 = 1.5f;
const float c2 = 1.5f;
// END: NEW PSO

double PSO_FUNC(double* P);

void initializeRandom();

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(const std::vector<float>& x);

struct Particle
{
  float NCC;
  std::vector<float> Position;
  std::vector<float> Velocity;

#ifdef Autoscoper_COLLISION_DETECTION
  bool Collided;
#endif // Autoscoper_COLLISION_DETECTION

  // Copy constructor
  Particle(const Particle& p);
  // Default constructor
  Particle();
  Particle(const std::vector<float>& pos);
  Particle(float start_range_min, float start_range_max);
  // Assignment operator
  Particle& operator=(const Particle& p);

  void updateVelocityAndPosition(const Particle& pBest, const Particle& gBest, float omega);
  void initializePosition(float start_range_min, float start_range_max);
};

// Stream operator
extern std::ostream& operator<<(std::ostream& os, const std::vector<float>& values);
extern std::ostream& operator<<(std::ostream& os, const Particle& p);

Particle pso(float start_range_min, float start_range_max, unsigned int MAX_EPOCHS, unsigned int MAX_STALL);

#ifdef Autoscoper_COLLISION_DETECTION
void checkCollision(Particle& p,
                    std::vector<float>& avgNonCollidedPosition,
                    std::vector<float>& avgCollidedPosition,
                    unsigned int& collidedCount);

void computeCorrectionVector(std::vector<float>& avgNonCollidedPosition,
                             std::vector<float>& avgCollidedPosition,
                             std::vector<float>& correctionVector,
                             const unsigned int& collidedCount);
#endif // Autoscoper_COLLISION_DETECTION
