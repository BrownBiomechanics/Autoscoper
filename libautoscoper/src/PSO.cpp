#include "PSO.hpp"
#include <iostream>
#include <string>
#include "PositionParticle.hpp"
#include "FilterParticle.hpp"

std::ostream& operator<<(std::ostream& os, const std::vector<float>& values)
{
  auto it = std::begin(values);
  for (auto value: values) {
    os << value;
    ++it;
    os << (it != std::end(values) ? ", " : "");
    }
  return os;
}

// New Particle Swarm Optimization
float host_fitness_function(Particle* p)
{
  // For position particles
  const PositionParticle* pos_p = dynamic_cast<PositionParticle*>(p);
  if (pos_p == nullptr) {
    std::cerr << "ERROR: p is not a PositionParticle" << std::endl;
    return FLT_MAX;
  }
  double xyzypr_manip[NUM_OF_POS_DIMENSIONS] = {0.0};
  for (int dim = 0; dim < NUM_OF_POS_DIMENSIONS; dim++) {
    xyzypr_manip[dim] = (double)pos_p->Position[dim];
  }

  double total = PSO_FUNC(xyzypr_manip);

  return (float)total;
}

float filter_fitness_function(Particle* p) {
  // For filter particles
  const FilterParticle* filter_p = dynamic_cast<FilterParticle*>(p);
  if (filter_p == nullptr) {
    std::cerr << "ERROR: p is not a FilterParticle" << std::endl;
    return FLT_MAX;
  }
  double filter_settings[NUM_OF_FILTER_DIMENSIONS] = {0.0};
  for (int dim = 0; dim < NUM_OF_FILTER_DIMENSIONS; dim++) {
    filter_settings[dim] = (double)filter_p->Filter_Settings[dim];
  }
  double total = FILTER_FUNC(filter_settings);
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

void intializePositionParticles(std::vector<Particle*>& particles, float start_range_min, float start_range_max)
{
  // First particle is the initial position
  particles[0] = new PositionParticle({ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f });
  for (int idx = 0; idx < NUM_OF_PARTICLES; idx++)
  {
    particles[idx] = new PositionParticle(start_range_min, start_range_max);
  }
}

void intializeFilterParticles(std::vector<Particle*>& particles, float start_range_min, float start_range_max)
{
  // First particle is the initial position
  particles[0] = new FilterParticle({ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
  for (int idx = 0; idx < NUM_OF_PARTICLES; idx++)
  {
    particles[idx] = new FilterParticle(start_range_min, start_range_max);
  }
}

Particle* pso(float start_range_min, float start_range_max, unsigned int MAX_EPOCHS, unsigned int MAX_STALL, bool position)
{
  int stall_iter = 0;
  bool do_this = true;
  unsigned int counter = 0;
  float OMEGA = 0.8f;

  // Pre-allocate particles
  std::vector<Particle*> particles(NUM_OF_PARTICLES);

  if (position)
    intializePositionParticles(particles, start_range_min, start_range_max);
  else
    intializeFilterParticles(particles, start_range_min, start_range_max);

  srand((unsigned)time(NULL));

  Particle* gBest;
  Particle* currentBest;
  if (position) {
    gBest = new PositionParticle();
    currentBest = new PositionParticle();
  }
  else {
    gBest = new FilterParticle();
    currentBest = new FilterParticle();
  }
  *gBest = *particles[0];

  // Make a copy of the particles, this will be the initial pBest
  std::vector<Particle*> pBest(NUM_OF_PARTICLES);
  for (int idx = 0; idx < NUM_OF_PARTICLES; idx++) {
    if (position)
      pBest[idx] = new PositionParticle(*dynamic_cast<PositionParticle*>(particles[idx]));
    else
      pBest[idx] = new FilterParticle(*dynamic_cast<FilterParticle*>(particles[idx]));
  }

  while (do_this)
  {
    //std::cout << "OMEGA: " << OMEGA << std::endl;
    if (counter >= MAX_EPOCHS) {
      do_this = false;
    }

    *currentBest = *gBest;

    for (int idx = 0; idx < NUM_OF_PARTICLES; idx++) {

      // Update the velocities and positions
      particles[idx]->updateParticle(*pBest[idx], *gBest, OMEGA);

      // Get the NCC of the current particle
      if (position)
        particles[idx]->NCC = host_fitness_function(particles[idx]);
      else
        particles[idx]->NCC = filter_fitness_function(particles[idx]);

      // Update the pBest if the current particle is better
      if (particles[idx]->NCC < pBest[idx]->NCC) {
        *pBest[idx] = *particles[idx];
      }

      // Update the gBest if the current particle is better
      if (particles[idx]->NCC < gBest->NCC) {
        *gBest = *particles[idx];
      }
    }

    OMEGA = OMEGA * 0.9f;

    std::cout << "Current Best NCC: " << gBest->NCC << std::endl;

    //std::cout << "Stall: " << stall_iter << std::endl;
    if (abs(gBest->NCC - currentBest->NCC) < 1e-4f) {
      //std::cout << "Increased Stall Iter" << std::endl;
      stall_iter++;
    } else if (abs(gBest->NCC - currentBest->NCC) > 0.001f) {
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

  return gBest;
}
