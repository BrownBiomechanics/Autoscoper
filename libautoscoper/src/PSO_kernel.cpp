#include "gpu/cuda/PSO_kernel.h"
#include <iostream>


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

float getRandom(float low, float high)
{
	return low + float(((high - low) + 1)*rand() / (RAND_MAX + 1.0));
}

float getRandomClamped()
{
	return (float)rand() / (float)RAND_MAX;
}

void pso(float *positions, float *velocities, float *pBests, float *gBest, unsigned int MAX_EPOCHS)
{
	int stall_iter = 0;
	float tempParticle1[NUM_OF_DIMENSIONS];
	float tempParticle2[NUM_OF_DIMENSIONS];

	bool do_this = true;
	unsigned int counter = 0;
	//for (int iter = 0; iter < (signed int)MAX_EPOCHS; iter++)

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

			positions[i] += velocities[i];
		}

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

		if (abs(epochBest - currentBest) < (float)1e-6)
		{
			stall_iter += 1;
		}
		if (stall_iter == 25)
		{
			std::cout << "Maximum Stall Iteration was reached" << std::endl;
			do_this = false;
		}

		counter += 1;
	}
	std::cout << "Total #Epoch of: " << counter << std::endl;
}
