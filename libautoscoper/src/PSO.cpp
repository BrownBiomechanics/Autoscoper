#include "PSO.hpp"
#include <iostream>
#include <cfloat> // For FLT_MAX
#include <string>

// Particle Struct Function Definitions
Particle::Particle(const Particle& p) {
  this->NCC = p.NCC;
  this->Position = p.Position;
  this->Velocity = p.Velocity;
}

Particle::Particle() {
  this->NCC = FLT_MAX;
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
}

Particle::Particle(const std::vector<float>& pos) {
  this->NCC = FLT_MAX;
  this->Position = pos;
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
}

Particle& Particle::operator=(const Particle& p) {
  this->NCC = p.NCC;
  this->Position = p.Position;
  this->Velocity = p.Velocity;
  return *this;
}

void Particle::updateVelocityAndPosition(Particle* pBest, Particle* gBest, float omega) {
  for (int dim = 0; dim < NUM_OF_DIMENSIONS; dim++) {
    float rp = getRandomClamped();
    float rg = getRandomClamped();

    this->Velocity.at(dim) =
        omega * this->Velocity.at(dim)
        + c1 * rp * (pBest->Position.at(dim) - this->Position.at(dim))
        + c2 * rg * (gBest->Position.at(dim) - this->Position.at(dim));

    this->Position.at(dim) += this->Velocity.at(dim);
  }
}

void Particle::initializePosition(float start_range_min, float start_range_max) {
  for (int dim = 0; dim < NUM_OF_DIMENSIONS; dim++) {
    this->Position.push_back(getRandom(start_range_min, start_range_max));
  }
}

// New Particle Swarm Optimization
float host_fitness_function(const std::vector<float>& x)
{
  double xyzypr_manip[6] = { 0.0 };
  for (int dim = 0; dim <= NUM_OF_DIMENSIONS - 1; dim++) {
    xyzypr_manip[dim] = (double)x[dim];
  }

  double total = PSO_FUNC(xyzypr_manip);

  return (float)total;
}

void intializeRandom()
{
  if (char* randomSeed = std::getenv("Autoscoper_RANDOM_SEED")) {
    try {
      std::cout << "Setting to Autoscoper_RANDOM_SEED to " << randomSeed << std::endl;
      unsigned int seed = std::stoi(std::string(randomSeed));
      srand(seed);
    }
    catch (const std::invalid_argument &e) {
      std::cerr << "Autoscoper_RANDOM_SEED is not a valid integer" << std::endl;
      exit(1);
    }
    catch (const std::out_of_range &e) {
      std::cerr << "Autoscoper_RANDOM_SEED is out of range" << std::endl;
      exit(1);
    }
  }
}

float getRandom(float low, float high)
{
  return low + getRandomClamped() * (high - low);
}

float getRandomClamped()
{
  return (float)rand() / (float)RAND_MAX;
}

void pso(std::vector<Particle>* particles, Particle* gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL)
{
  int stall_iter = 0;
  bool do_this = true;
  unsigned int counter = 0;
  float OMEGA = 0.8f;

  // Make a copy of the particles, this will be the initial pBest
  std::vector<Particle> pBest;
  Particle pBestTemp;
  for (const Particle& p : *particles) {
    pBestTemp = p;
    pBest.push_back(pBestTemp);
  }

  // Calc NCC for gBest
  gBest->NCC = host_fitness_function(gBest->Position);

  Particle currentBest;
  while (do_this)
  {
    //std::cout << "OMEGA: " << OMEGA << std::endl;
    if (counter >= MAX_EPOCHS) {
      do_this = false;
    }

    currentBest = *gBest;
    for (int idx = 0; idx < NUM_OF_PARTICLES; idx++) {

      // Update the velocities and positions
      particles->at(idx).updateVelocityAndPosition(&pBest.at(idx), gBest, OMEGA);

      // Get the NCC of the current particle
      particles->at(idx).NCC = host_fitness_function(particles->at(idx).Position);

      // Update the pBest if the current particle is better
      if (particles->at(idx).NCC < pBest.at(idx).NCC) {
        pBest.at(idx) = particles->at(idx);
      }

      // Update the gBest if the current particle is better
      if (particles->at(idx).NCC < gBest->NCC) {
        *gBest = particles->at(idx);
      }
    }

    OMEGA = OMEGA * 0.9f;

    std::cout << "Current Best NCC: " << gBest->NCC << std::endl;
    //std::cout << "Stall: " << stall_iter << std::endl;
    if (abs(gBest->NCC - currentBest.NCC) < 1e-4f) {
      //std::cout << "Increased Stall Iter" << std::endl;
      stall_iter++;
    } else if (abs(gBest->NCC - currentBest.NCC) > 0.001f) {
      //std::cout << "Zeroed Stall Iter" << std::endl;
      stall_iter = 0;
    }
    if (stall_iter == MAX_STALL) {
      std::cout << "Maximum Stall Iteration was reached" << std::endl;
      do_this = false;
    }

    counter++;
  }
  std::cout << "Total #Epoch of: " << counter << std::endl;
}
