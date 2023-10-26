#include "PSO.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

// Prevents multiple threads from writing to the same memory location
std::mutex MTX;

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

void thread_handler(Particle* p, Particle* pBest, Particle* gBest, float OMEGA) {
  // Update the velocities and positions
  p->updateVelocityAndPosition(pBest, gBest, OMEGA);

  // Get the NCC of the current particle
  p->ncc_val = host_fitness_function(p->position);

  // Update the pBest if the current particle is better
  if (p->ncc_val < pBest->ncc_val) {
    *pBest = *p;
  }

  // Critial Section
  MTX.lock();
  // Update the gBest if the current particle is better
  if (p->ncc_val < gBest->ncc_val) {
    *gBest = *p;
  }
  MTX.unlock();
}

void pso(std::vector<Particle*>* particles, Particle* gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL)
{
  int stall_iter = 0;
  bool do_this = true;
  unsigned int counter = 0;
  float OMEGA = 0.8f;

  // Make a copy of the particles, this will be the initial pBest
  std::vector<Particle*> pBest;
  Particle* pBestTemp = new Particle();
  for (Particle* p : *particles) {
    *pBestTemp = *p;
    pBest.push_back(pBestTemp);
  }
  pBestTemp = nullptr;
  delete pBestTemp;

  // Calc NCC for gBest
  gBest->ncc_val = host_fitness_function(gBest->position);

  // Init pointers
  Particle* currentBest = new Particle();
  Particle* p = new Particle();
  Particle* curPBest = new Particle();
  while (do_this)
  {
    //std::cout << "OMEGA: " << OMEGA << std::endl;
    if (counter >= MAX_EPOCHS)
    {
      do_this = false;
    }

    *currentBest = *gBest; // We want this to be a copy not a pointer

    // Create a thread for each particle
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_OF_PARTICLES; i++)
    {
      p = particles->at(i); // We want these to be pointers not copies
      curPBest = pBest.at(i);

      threads.push_back(std::thread(thread_handler, p, curPBest, currentBest, OMEGA));
    }

    // Wait for all threads to finish
    for (auto& t : threads) t.join();

    // Update the OMEGA
    OMEGA = OMEGA * 0.9f;

    std::cout << "Current Best NCC: " << gBest->ncc_val << std::endl;
    //std::cout << "Stall: " << stall_iter << std::endl;
    if (abs(gBest->ncc_val - currentBest->ncc_val) < 1e-4f)
    {
      //std::cout << "Increased Stall Iter" << std::endl;
      stall_iter++;
    } else if (abs(gBest->ncc_val - currentBest->ncc_val) > 0.001f)
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
  // Clean up pointers
  p = nullptr;
  delete p;
  curPBest = nullptr;
  delete curPBest;
  currentBest = nullptr;
  delete currentBest;

  std::cout << "Total #Epoch of: " << counter << std::endl;
}
