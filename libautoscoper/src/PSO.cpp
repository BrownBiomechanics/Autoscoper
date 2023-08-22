#include "PSO.hpp"
#include <iostream>
#include <string>

#define VELOCITY_FILTER 0
#define COLLISION_RESPONSE 0

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

  float velClamp = 0.2;

  // Inertial weight of PSO. Measure of how likley a particle is to remain
  // traveling in the same direciton it has been traveling (momentum like)
  float OMEGA = 0.8f;

  float OMEGA_MIN = 0.0001; // Note: Currently unused. Needed for adaptive PSO

  // Arrays for collision response
  float avgNonCollidedPosition[NUM_OF_DIMENSIONS];
  float avgCollidedPosition[NUM_OF_DIMENSIONS];
  float correctionVec[NUM_OF_DIMENSIONS];
  bool collided[NUM_OF_PARTICLES];

  while (do_this)
  {
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
#if VELOCITY_FILTER
      float speed = 0.0;
      for(int i=0; i< NUM_OF_DIMENSIONS; i++){
        speed += velocities[i] * velocities[i];
      }
      speed = sqrt(speed);
      if (speed > velClamp) {
        for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
          velocities[i] /= speed;
          velocities[i] *= velClamp;
        }
      }
#endif


      positions[i] += velocities[i];
    }
    // Note: For possible better performance, consider changing this to be linearly decreasing, with a floor at w=0.4 (DOI: 10.1080/0952813X.2013.782348)
    OMEGA = OMEGA * 0.9f;

    float globalBest = currentBest;
#if COLLISION_RESPONSE
    int collidedCount = 0;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
      avgNonCollidedPosition[j] = 0.0;
      avgCollidedPosition[j] = 0.0;
    }
    
    for (int j = 0; j < NUM_OF_PARTICLES; j++)
    {
      collided[j] = false;
    }
#endif

    int particleId = 0;
    for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i += NUM_OF_DIMENSIONS)
    {
      for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
      {
        tempParticle1[j] = positions[i + j];
        tempParticle2[j] = pBests[i + j];
      }

      // Normalized cross correlation for this particles current position
      float nccP = host_fitness_function(tempParticle1);

#if COLLISION_RESPONSE
      if (nccP > 1.0E4) {
        collidedCount++;

        collided[particleId] = true;
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
          avgCollidedPosition[j] += tempParticle1[j];
        }
      }
      
      if (nccP < 1.0E4){
        
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
          avgNonCollidedPosition[j] += tempParticle1[j];
        }
      }
#endif

      // Normalized cross correlation for this particle in its best position so far
      float nccPBest = host_fitness_function(tempParticle2);

      if (nccP < nccPBest)
      {
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
        {
          pBests[i + j] = positions[i + j];
        }
        if (nccP < globalBest)
        {
          //cout << "Current Best is: " ;
          for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
          {
            gBest[j] = pBests[i + j];
          }
          globalBest = nccP;
        }
      }
      particleId++;
    }
#if COLLISION_RESPONSE
    if (collidedCount != 0 && collidedCount != 100) {

      for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {


        avgNonCollidedPosition[j] /= (float)NUM_OF_PARTICLES - (float)collidedCount;
        avgCollidedPosition[j] /= (float)collidedCount;
        
        correctionVec[j] = avgNonCollidedPosition[j] - avgCollidedPosition[j];

      }
    }

    if (collidedCount != 0 && collidedCount != 100) {
      particleId = 0;
      for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i += NUM_OF_DIMENSIONS)
      {
        if (collided[particleId] == true) {
#if DEBUG
          std::cout << "Shifted particle = " << particleId << std::endl;
#endif
          for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
          {
            positions[i + j] += ((float)collidedCount/(float)NUM_OF_PARTICLES)*correctionVec[j];
            pBests[i + j] = positions[i + j];
          }
        }

        particleId++;
      }
    }
#endif

    float epochBest = globalBest; 
    // float epochBest = host_fitness_function(gBest);

    
    
    if (counter % 5 == 0) {
      std::cout << "Current Best NCC: " << epochBest
        << "   on iteration:" << counter
        << " and stall = " << stall_iter << std::endl;
    }
    if (abs(epochBest - currentBest) < 1e-4f)
    {
      //std::cout << "Increased Stall Iter" << std::endl;
      stall_iter++;
    } 
    else if (abs(epochBest - currentBest) > 0.001f)
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
