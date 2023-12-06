#include "FilterParticle.hpp"

// Default constructor
FilterParticle::FilterParticle() : Particle(8) {
  this->Filter_Settings = std::vector<float>(this->NUM_OF_DIMENSIONS, 0.f);
  this->Velocity = std::vector<float>(this->NUM_OF_DIMENSIONS, 0.f);
}

// Copy constructor
FilterParticle::FilterParticle(const FilterParticle& p) : Particle(8) {
  this->Filter_Settings = p.Filter_Settings;
  this->Velocity = p.Velocity;
  this->NCC = p.NCC;
}

// Filter settings constructor
FilterParticle::FilterParticle(const std::vector<float>& filter_settings) : Particle(8) {
  this->Filter_Settings = filter_settings;
  this->Velocity = std::vector<float>(this->NUM_OF_DIMENSIONS, 0.f);
}

// Constructor with rand range
FilterParticle::FilterParticle(float start_range_min, float start_range_max) : Particle(8) {
  this->Filter_Settings = std::vector<float>(this->NUM_OF_DIMENSIONS, 0.f);
  this->Velocity = std::vector<float>(this->NUM_OF_DIMENSIONS, 0.f);
  for (int i = 0; i < this->NUM_OF_DIMENSIONS; i++) {
    this->Filter_Settings[i] = getRandom(start_range_min, start_range_max);
  }
}

void FilterParticle::updateParticle(const Particle& pBest, const Particle& gBest, float omega) {
  // Check that the particles are the same type
  const FilterParticle* p = dynamic_cast<const FilterParticle*>(&pBest);
  const FilterParticle* g = dynamic_cast<const FilterParticle*>(&gBest);
  if (p == nullptr || g == nullptr) {
    std::cout << "Error: Either pBest or gBest is not a FilterParticle" << std::endl;
    return;
  }
  for (int dim = 0; dim < this->NUM_OF_DIMENSIONS; dim++) {
    float rp = getRandomClamped();
    float rg = getRandomClamped();

    this->Velocity[dim] =
      omega * this->Velocity[dim]
      + c1 * rp * (p->Filter_Settings[dim] - this->Filter_Settings[dim])
      + c2 * rg * (g->Filter_Settings[dim] - this->Filter_Settings[dim]);

    this->Filter_Settings[dim] += this->Velocity[dim];
  }
}

// Assignment operator
Particle& FilterParticle::operator=(const Particle& p) {
  const FilterParticle* fp = dynamic_cast<const FilterParticle*>(&p);
  if (fp == nullptr) {
    std::cerr << "ERROR: p is not a FilterParticle" << std::endl;
    return *this;
  }
  this->Filter_Settings = fp->Filter_Settings;
  this->Velocity = fp->Velocity;
  this->NCC = fp->NCC;
  return *this;
}

// Output operator
std::ostream& operator<<(std::ostream& os, const FilterParticle& p) {
  os << "Filter Settings: " << p.Filter_Settings << std::endl;
  os << "Velocity: " << p.Velocity << std::endl;
  os << "NCC: " << p.NCC;
  return os;
}
