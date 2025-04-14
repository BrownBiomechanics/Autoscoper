#include "PSO.hpp"
#include <iostream>
#include <cfloat> // For FLT_MAX
#include <string>

#define VELOCITY_FILTER 0
#define COLLISION_RESPONSE 0

// Particle Struct Function Definitions
Particle::Particle(const Particle& p)
{
  this->NCC = p.NCC;
  this->Position = p.Position;
  this->Velocity = p.Velocity;
#ifdef Autoscoper_COLLISION_DETECTION
  this->Collided = p.Collided;
#endif // Autoscoper_COLLISION_DETECTION
}

Particle::Particle()
{
  this->NCC = FLT_MAX;
  this->Position = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);

#ifdef Autoscoper_COLLISION_DETECTION
  this->Collided = false;
#endif // Autoscoper_COLLISION_DETECTION
}

Particle::Particle(const std::vector<float>& pos)
{
  this->NCC = FLT_MAX;
  this->Position = pos;
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);

#ifdef Autoscoper_COLLISION_DETECTION
  this->Collided = false;
#endif // Autoscoper_COLLISION_DETECTION
}

Particle::Particle(float start_range_min, float start_range_max)
{
  this->NCC = FLT_MAX;
  this->Position = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
  this->Velocity = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
  this->initializePosition(start_range_min, start_range_max);

#ifdef Autoscoper_COLLISION_DETECTION
  this->Collided = false;
#endif // Autoscoper_COLLISION_DETECTION
}

Particle& Particle::operator=(const Particle& p)
{
  this->NCC = p.NCC;
  this->Position = p.Position;
  this->Velocity = p.Velocity;

#ifdef Autoscoper_COLLISION_DETECTION
  this->Collided = p.Collided;
#endif // Autoscoper_COLLISION_DETECTION

  return *this;
}

std::ostream& operator<<(std::ostream& os, const std::vector<float>& values)
{
  auto it = std::begin(values);
  for (auto value : values) {
    os << value;
    ++it;
    os << (it != std::end(values) ? ", " : "");
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Particle& p)
{
  os << "Position: " << p.Position << std::endl;
  os << "Velocity: " << p.Velocity << std::endl;
  os << "NCC: " << p.NCC;
  return os;
}

void Particle::updateVelocityAndPosition(const Particle& pBest, const Particle& gBest, float omega)
{
  for (int dim = 0; dim < NUM_OF_DIMENSIONS; dim++) {
    float rp = getRandomClamped();
    float rg = getRandomClamped();

    this->Velocity[dim] = omega * this->Velocity[dim] + c1 * rp * (pBest.Position[dim] - this->Position[dim])
                          + c2 * rg * (gBest.Position[dim] - this->Position[dim]);

#ifdef VELOCITY_FILTER
    float velClamp = 0.2;
    float speed = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
      speed += this->Velocity[i] * this->Velocity[i];
    }
    speed = sqrt(speed);
    if (speed > velClamp) {
      for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
        this->Velocity[i] /= speed;
        this->Velocity[i] *= velClamp;
      }
    }
#endif // VELOCITY_FILTER

    this->Position[dim] += this->Velocity[dim];
  }
}

void Particle::initializePosition(float start_range_min, float start_range_max)
{
  for (int dim = 0; dim < NUM_OF_DIMENSIONS; dim++) {
    this->Position[dim] = getRandom(start_range_min, start_range_max);
  }
}

// New Particle Swarm Optimization
float host_fitness_function(const std::vector<float>& x)
{
  double xyzypr_manip[NUM_OF_DIMENSIONS] = { 0.0 };
  for (int dim = 0; dim < NUM_OF_DIMENSIONS; dim++) {
    xyzypr_manip[dim] = (double)x[dim];
  }

  double total = PSO_FUNC(xyzypr_manip);

  return (float)total;
}

void initializeRandom()
{
  unsigned int seed = (unsigned)time(NULL);

  if (char* randomSeed = std::getenv("Autoscoper_RANDOM_SEED")) {
    try {
      std::cout << "Autoscoper_RANDOM_SEED env. variable is set to " << randomSeed << std::endl;
      seed = std::stoi(std::string(randomSeed));
    } catch (const std::invalid_argument& e) {
      std::cerr << "Autoscoper_RANDOM_SEED is not a valid integer" << std::endl;
      exit(1);
    } catch (const std::out_of_range& e) {
      std::cerr << "Autoscoper_RANDOM_SEED is out of range" << std::endl;
      exit(1);
    }
  }
  std::cout << "Initializing Autoscoper random number generator with " << seed << std::endl;
  srand(seed);
}

float getRandom(float low, float high)
{
  return low + getRandomClamped() * (high - low);
}

float getRandomClamped()
{
  return (float)rand() / (float)RAND_MAX;
}

#ifdef Autoscoper_COLLISION_DETECTION
void checkCollision(Particle& p,
                    std::vector<float>& avgNonCollidedPosition,
                    std::vector<float>& avgCollidedPosition,
                    unsigned int& collidedCount)
{
  if (p.NCC > 1.0E4) {
    collidedCount++;
    p.Collided = true;

    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
      avgCollidedPosition[i] += p.Position[i];
    }
  } else {
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
      avgNonCollidedPosition[i] += p.Position[i];
    }
  }
}

void computeCorrectionVector(std::vector<float>& avgNonCollidedPosition,
                             std::vector<float>& avgCollidedPosition,
                             std::vector<float>& correctionVector,
                             const unsigned int& collidedCount)
{
  if (collidedCount != 0 && collidedCount != NUM_OF_PARTICLES) {
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
      avgNonCollidedPosition[i] /= ((float)NUM_OF_PARTICLES - (float)collidedCount);
      avgCollidedPosition[i] /= (float)collidedCount;
    }
  }
}
#endif // Autoscoper_COLLISION_DETECTION

Particle pso(float start_range_min, float start_range_max, unsigned int MAX_EPOCHS, unsigned int MAX_STALL)
{
  int stall_iter = 0;
  bool do_this = true;
  unsigned int counter = 0;
  float OMEGA = 0.8f;

  // Pre-allocate particles
  std::vector<Particle> particles(NUM_OF_PARTICLES);

  // First particle is the initial position
  particles[0] = Particle({ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f });

  // ... and the other particles positions are randomly iniialized
  for (int idx = 1; idx < NUM_OF_PARTICLES; idx++) {
    particles[idx] = Particle(start_range_min, start_range_max);
  }

  Particle gBest;

  // Make a copy of the particles, this will be the initial pBest
  std::vector<Particle> pBest = particles;

  Particle currentBest;
  while (do_this) {
    // std::cout << "OMEGA: " << OMEGA << std::endl;
    if (counter >= MAX_EPOCHS) {
      do_this = false;
    }

    currentBest = gBest;
#ifdef Autoscoper_COLLISION_DETECTION
    unsigned int collidedCount = 0;
    std::vector<float> avgNonCollidedPosition = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
    std::vector<float> avgCollidedPosition = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
    std::vector<float> correctionVector = std::vector<float>(NUM_OF_DIMENSIONS, 0.f);
#endif // Autoscoper_COLLISION_DETECTION

    for (int idx = 0; idx < NUM_OF_PARTICLES; idx++) {

      // Update the velocities and positions
      particles[idx].updateVelocityAndPosition(pBest[idx], gBest, OMEGA);

      // Get the NCC of the current particle
      particles[idx].NCC = host_fitness_function(particles[idx].Position);

      // Update the pBest if the current particle is better
      if (particles[idx].NCC < pBest[idx].NCC) {
        pBest[idx] = particles[idx];
      }

      // Update the gBest if the current particle is better
      if (particles[idx].NCC < gBest.NCC) {
        gBest = particles[idx];
      }
#ifdef Autoscoper_COLLISION_DETECTION
      checkCollision(particles[idx], avgNonCollidedPosition, avgCollidedPosition, collidedCount);
#endif // Autoscoper_COLLISION_DETECTION
    }

#if defined(Autoscoper_COLLISION_DETECTION) and defined(COLLISION_RESPONSE)
    computeCorrectionVector(avgNonCollidedPosition, avgCollidedPosition, correctionVector, collidedCount);
    for (int idx = 0; idx < NUM_OF_PARTICLES; ++idx) {
      if (particles[idx].Collided) {
        for (int j = 0; j < NUM_OF_DIMENSIONS; ++j) {
          particles[idx].Position[j] += ((float)collidedCount / (float)NUM_OF_PARTICLES) * correctionVector[j];
          pBest[idx] = particles[idx];
          // Why always update the pBest here?
        }
      }
    }
    if (collidedCount) {
      std::cout << "PSO: epoch #" << counter << ": total collided particles: " << collidedCount << std::endl;
    }
#endif // Autoscoper_COLLISION_DETECTION and COLLISION_RESPONSE

    OMEGA = OMEGA * 0.9f;

    std::cout << "Current Best NCC: " << gBest.NCC << std::endl;

    // std::cout << "Stall: " << stall_iter << std::endl;
    if (abs(gBest.NCC - currentBest.NCC) < 1e-4f) {
      // std::cout << "Increased Stall Iter" << std::endl;
      stall_iter++;
    } else if (abs(gBest.NCC - currentBest.NCC) > 0.001f) {
      // std::cout << "Zeroed Stall Iter" << std::endl;
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
