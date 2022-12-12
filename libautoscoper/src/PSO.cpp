#include "PSO.hpp"
#include <iostream>
#include <string>


// New Particle Swarm Optimization
float host_fitness_function(float x[])
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
  return low + (high - low + 1.0f)*(float)rand() / (RAND_MAX + 1.0f);
}

float getRandomClamped()
{
  return (float)rand() / (float)RAND_MAX;
}

void pso(float *positions, float *velocities, float *pBests, float *gBest, unsigned int MAX_EPOCHS, unsigned int MAX_STALL)
{
  int stall_iter = 0;
  float tempParticle1[NUM_OF_DIMENSIONS];
  float tempParticle2[NUM_OF_DIMENSIONS];

  bool do_this = true;
  unsigned int counter = 0;
  //for (int iter = 0; iter < (signed int)MAX_EPOCHS; iter++)

  float OMEGA = 0.8f;

  while (do_this)
  {
    //std::cout << "OMEGA: " << OMEGA << std::endl;
    if (counter >= MAX_EPOCHS)
    {
      do_this = false;
    }

    float currentBest = host_fitness_function(gBest);

    for (int i = 0; i < NUM_OF_PARTICLES*NUM_OF_DIMENSIONS; i++)
    {
      float rp = getRandomClamped();
      float rg = getRandomClamped();

      velocities[i] = OMEGA * velocities[i] + c1 * rp*(pBests[i] - positions[i]) + c2 * rg*(gBest[i%NUM_OF_DIMENSIONS] - positions[i]);

      positions[i] += velocities[i];
    }

    OMEGA = OMEGA * 0.9f;

    for (int i = 0; i < NUM_OF_PARTICLES*NUM_OF_DIMENSIONS; i += NUM_OF_DIMENSIONS)
    {
      for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
      {
        tempParticle1[j] = positions[i + j];
        tempParticle2[j] = pBests[i + j];
      }

      if (host_fitness_function(tempParticle1) < host_fitness_function(tempParticle2))
      {
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
        {
          pBests[i + j] = positions[i + j];
        }

        if (host_fitness_function(tempParticle1) < host_fitness_function(gBest))
        {
          //cout << "Current Best is: " ;
          for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
          {
            gBest[j] = pBests[i + j];
          }
        }
      }
    }

    float epochBest = host_fitness_function(gBest);

    std::cout << "Current Best NCC: " << epochBest << std::endl;
    //std::cout << "Stall: " << stall_iter << std::endl;
    if (abs(epochBest - currentBest) < 1e-4f)
    {
      //std::cout << "Increased Stall Iter" << std::endl;
      stall_iter++;
    } else if (abs(epochBest - currentBest) > 0.001f)
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
