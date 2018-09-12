#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "PSO_kernel.h"

__device__ float fitness_function(float x[])
{
	double xyzypr_manip[6] = { 0 };
	for (int i = 0; i <= NUM_OF_DIMENSIONS - 1; i++)
	{
		xyzypr_manip[i] = (double)x[i];
	} // i

	double total = PSO_FUNC(xyzypr_manip);

	//cout << "Check total function: " << total << endl;
	return (float)total;
}

__global__ void kernelUpdateParticle(float *positions, float *velocities, float *pBests, float *gBest, float r1, float r2)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
		return;

	float rp = r1;
	float rg = r2;

	velocities[i] = OMEGA * velocities[i] + c1 * rp*(pBests[i] - positions[i]) + c2 * rg*(gBest[i%NUM_OF_DIMENSIONS] - positions[i]);
	positions[i] += velocities[i];
}

__global__ void kernelUpdatePBest(float *positions, float *pBests, float *gBest)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
		return;

	float tempParticle1[NUM_OF_DIMENSIONS];
	float tempParticle2[NUM_OF_DIMENSIONS];

	for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
	{
		tempParticle1[j] = positions[i + j];
		tempParticle2[j] = pBests[i + j];
	}

	if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
	{
		for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
			pBests[i + j] = tempParticle1[j];

		if (fitness_function(tempParticle1) < fitness_function(gBest))
		{
			for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
				atomicExch(gBest + j, tempParticle1[j]);
		}
	}
}

__global__ void kernelUpdateGBest(float *gBest, float *pBests)
{
	float temp[NUM_OF_DIMENSIONS];
	for (int i = 0; i < 10 * NUM_OF_DIMENSIONS; i += NUM_OF_DIMENSIONS)
	{
		for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
			temp[k] = pBests[i + k];

		if (fitness_function(temp) < fitness_function(gBest))
		{
			for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
				gBest[k] = temp[k];
		}
	}
}


extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest, unsigned int MAX_EPOCHS)
{
	int size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;

	float *devPos;
	float *devVel;
	float *devPBest;
	float *devGBest;


	/*cudaDeviceProp prop;
	int deviceNum;
	cudaGetDeviceCount(&deviceNum);
	for(int i=0;i<deviceNum;i++)
	{
		cudaGetDeviceProperties(&prop,i);

		if(!prop.deviceOverlap)
		{
			printf("No device will handle overlaps, so no speed up from stream.\n");
		}
	}*/

	cudaMalloc((void**)&devPos, sizeof(float)*size);
	cudaMalloc((void**)&devVel, sizeof(float)*size);
	cudaMalloc((void**)&devPBest, sizeof(float)*size);
	cudaMalloc((void**)&devGBest, sizeof(float)*NUM_OF_DIMENSIONS);

	int threadNum = 64;
	int blocksNum = NUM_OF_PARTICLES / threadNum;

	cudaMemcpy(devPos, positions, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVel, velocities, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(devPBest, pBests, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(devGBest, gBest, sizeof(float)*NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);

	//cudaEvent_t start1;
	//cudaEventCreate(&start1);
	//cudaEvent_t stop1;
	//cudaEventCreate(&stop1);
	//float msecTotal1 = 0.0f;
	for (int iter = 0; iter < MAX_EPOCHS; iter++)
	{
		kernelUpdateParticle << <blocksNum, threadNum >> > (devPos, devVel, devPBest, devGBest, getRandomClamped(), getRandomClamped());//0.000008s

		//cudaEventRecord(start1, NULL);

		kernelUpdatePBest << <blocksNum, threadNum >> > (devPos, devPBest, devGBest);

		//cudaEventRecord(stop1, NULL);
		//cudaEventSynchronize(stop1);
		//cudaEventElapsedTime(&msecTotal1, start1, stop1);
		//printf("Time elapsed:%10.10lf s\n",(double)msecTotal1/1000);
	}

	cudaMemcpy(positions, devPos, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities, devVel, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(pBests, devPBest, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(gBest, devGBest, sizeof(float)*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);

	cudaFree(devPos);
	cudaFree(devVel);
	cudaFree(devPBest);
	cudaFree(devGBest);
}
