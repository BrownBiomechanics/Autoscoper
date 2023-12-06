#include "PositionParticle.hpp"

// PositionParticle Struct Function Definitions
PositionParticle::PositionParticle(const PositionParticle& p) : Particle(6) {
  this->NCC = p.NCC;
  this->Position = p.Position;
  this->Velocity = p.Velocity;
}

PositionParticle::PositionParticle() : Particle(6) {
  this->Position = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
}

PositionParticle::PositionParticle(const std::vector<float>& pos) : Particle(6) {
  this->Position = pos;
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
}

PositionParticle::PositionParticle(float start_range_min, float start_range_max) : Particle(6) {
  this->Position = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
  this->initializePosition(start_range_min, start_range_max);
}

Particle& PositionParticle::operator=(const Particle& p) {
  const PositionParticle* posP = dynamic_cast<const PositionParticle*>(&p);
  if (posP == nullptr) {
    std::cerr << "ERROR: p is not a PositionParticle" << std::endl;
    return *this;
  }
  this->NCC = posP->NCC;
  this->Position = posP->Position;
  this->Velocity = posP->Velocity;
  return *this;
}

void PositionParticle::updateParticle(const Particle& pBest, const Particle& gBest, float omega) {
  // Check if pBest and gBest are PositionParticles
  const PositionParticle* pBestPos = dynamic_cast<const PositionParticle*>(&pBest);
  const PositionParticle* gBestPos = dynamic_cast<const PositionParticle*>(&gBest);
  if (pBestPos == nullptr || gBestPos == nullptr) {
    std::cerr << "ERROR: pBest or gBest is not a PositionParticle" << std::endl;
    return;
  }
  for (int dim = 0; dim < NUM_OF_DIMENSIONS; dim++) {
    float rp = getRandomClamped();
    float rg = getRandomClamped();

    this->Velocity[dim] =
      omega * this->Velocity[dim]
      + c1 * rp * (pBestPos->Position[dim] - this->Position[dim])
      + c2 * rg * (gBestPos->Position[dim] - this->Position[dim]);

    this->Position[dim] += this->Velocity[dim];
  }
}

void PositionParticle::initializePosition(float start_range_min, float start_range_max) {
  for (int dim = 0; dim < NUM_OF_DIMENSIONS; dim++) {
    this->Position[dim] = getRandom(start_range_min, start_range_max);
  }
}

std::ostream& operator<<(std::ostream& os, const PositionParticle& p)
{
  os << "Position: " << p.Position << std::endl;
  os << "Velocity: " << p.Velocity << std::endl;
  os << "NCC: " << p.NCC;
  return os;
}