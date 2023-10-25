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
#include <vector>
const int NUM_OF_PARTICLES = 100;
const int NUM_OF_DIMENSIONS = 6;
const float c1 = 1.5f;
const float c2 = 1.5f;
// END: NEW PSO

double PSO_FUNC(double *P);

void intializeRandom();

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(std::vector<float> x);

struct Particle {
  float ncc_val;
  std::vector<float> position;
  std::vector<float> velocity;

  // Copy constructor
  Particle(const Particle& p) {
    ncc_val = p.ncc_val;
    position = p.position;
    velocity = p.velocity;
  }
  // Default constructor
  Particle() {
    ncc_val = FLT_MAX;
    velocity = *(new std::vector<float>(NUM_OF_DIMENSIONS, 0.f));
  }
  Particle(const std::vector<float>& pos) {
    ncc_val = FLT_MAX;
    position = pos;
    velocity = *(new std::vector<float>(NUM_OF_DIMENSIONS, 0.f));
  }
  // Assignment operator
  Particle& operator=(const Particle& p) {
    ncc_val = p.ncc_val;
    position = p.position;
    velocity = p.velocity;
    return *this;
  }
  void updateVelocityAndPosition(Particle* pBest, Particle* gBest, float OMEGA) {
    for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
      float rp = getRandomClamped();
      float rg = getRandomClamped();

      this->velocity.at(i) = OMEGA * velocity.at(i) + c1 * rp * (pBest->position.at(i) - this->position.at(i)) + c2 * rg * (gBest->position.at(i) - this->position.at(i));
      this->position.at(i) += this->velocity.at(i);
    }
  }
};

void pso(std::vector<Particle*>* particles, Particle* gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL);
