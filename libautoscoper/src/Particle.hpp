#pragma once
#include <cfloat> // For FLT_MAX
#include <vector>
#include <iostream>

const float c1 = 1.5f;
const float c2 = 1.5f;

struct Particle {
  float NCC;
  std::vector<float> Velocity;

  virtual void updateParticle(const Particle& pBest, const Particle& gBest, float omega) = 0;

  // Assignment operator
  virtual Particle& operator=(const Particle& p) = 0;

  float getRandomClamped()
  {
    return (float)rand() / (float)RAND_MAX;
  }

  float getRandom(float low, float high)
  {
    return low + getRandomClamped() * (high - low);
  }
};
