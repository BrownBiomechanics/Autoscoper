#include "PSO.hpp"
#include <iostream>
#include <string>

// Particle Struct Function Definitions
Particle::Particle(const Particle& p) {
  ncc_val = p.ncc_val;
  position = p.position;
  velocity = p.velocity;
}

Particle::Particle() {
  ncc_val = FLT_MAX;
  velocity = *(new std::vector<float>(NUM_OF_DIMENSIONS, 0.f));
}

Particle::Particle(const std::vector<float>& pos) {
  ncc_val = FLT_MAX;
  position = pos;
  velocity = *(new std::vector<float>(NUM_OF_DIMENSIONS, 0.f));
}

Particle& Particle::operator=(const Particle& p) {
  ncc_val = p.ncc_val;
  position = p.position;
  velocity = p.velocity;
  return *this;
}

void Particle::updateVelocityAndPosition(Particle* pBest, Particle* gBest, float OMEGA) {
  for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
    float rp = getRandomClamped();
    float rg = getRandomClamped();

    this->velocity.at(i) = OMEGA * velocity.at(i) + c1 * rp * (pBest->position.at(i) - this->position.at(i)) + c2 * rg * (gBest->position.at(i) - this->position.at(i));
    this->position.at(i) += this->velocity.at(i);
  }
}

void Particle::InitializePosition(float START_RANGE_MIN, float START_RANGE_MAX) {
  for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
    this->position.push_back(getRandom(START_RANGE_MIN, START_RANGE_MAX));
  }
}

// New Particle Swarm Optimization
float host_fitness_function(std::vector<float> x)
{
  double xyzypr_manip[6] = { 0 };
  for (int i = 0; i <= NUM_OF_DIMENSIONS - 1; i++)
  {
    xyzypr_manip[i] = (double)x[i];
  } // i

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
  for (Particle p : *particles) {
    pBestTemp = p;
    pBest.push_back(pBestTemp);
  }

  // Calc NCC for gBest
  gBest->ncc_val = host_fitness_function(gBest->position);

  Particle currentBest;
  while (do_this)
  {
    //std::cout << "OMEGA: " << OMEGA << std::endl;
    if (counter >= MAX_EPOCHS)
    {
      do_this = false;
    }

    currentBest = *gBest;

    for (int i = 0; i < NUM_OF_PARTICLES; i++)
    {

      // Update the velocities and positions
      particles->at(i).updateVelocityAndPosition(&pBest.at(i), gBest, OMEGA);

      // Get the NCC of the current particle
      particles->at(i).ncc_val = host_fitness_function(particles->at(i).position);

      // Update the pBest if the current particle is better
      if (particles->at(i).ncc_val < pBest.at(i).ncc_val) {
        pBest.at(i) = particles->at(i);
      }

      // Update the gBest if the current particle is better
      if (particles->at(i).ncc_val < gBest->ncc_val) {
        *gBest = particles->at(i);
      }
    }

    OMEGA = OMEGA * 0.9f;

    std::cout << "Current Best NCC: " << gBest->ncc_val << std::endl;
    //std::cout << "Stall: " << stall_iter << std::endl;
    if (abs(gBest->ncc_val - currentBest.ncc_val) < 1e-4f)
    {
      //std::cout << "Increased Stall Iter" << std::endl;
      stall_iter++;
    } else if (abs(gBest->ncc_val - currentBest.ncc_val) > 0.001f)
    {
      //std::cout << "Zeroed Stall Iter" << std::endl;
      stall_iter = 0;
    }
    if (stall_iter == MAX_STALL)
    {
      std::cout << "Maximum Stall Iteration was reached" << std::endl;
      do_this = false;
    }

    counter++;
  }
  std::cout << "Total #Epoch of: " << counter << std::endl;
}
