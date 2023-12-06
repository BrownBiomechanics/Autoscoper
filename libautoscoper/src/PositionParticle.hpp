#pragma once
#include "Particle.hpp"

struct PositionParticle : public Particle {
  std::vector<float> Position;

  // Copy constructor
  PositionParticle(const PositionParticle& p);
  // Default constructor
  PositionParticle();
  PositionParticle(const std::vector<float>& pos);
  PositionParticle(float start_range_min, float start_range_max);
  // Assignment operator
  Particle& operator=(const Particle& p);
  void updateParticle(const Particle& pBest, const Particle& gBest, float omega);
  void initializePosition(float start_range_min, float start_range_max);
};

extern std::ostream& operator<<(std::ostream& os, const PositionParticle& p);
