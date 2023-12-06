#include "Particle.hpp"
#include "View.hpp"

struct FilterParticle : public Particle {
  std::vector<float> Filter_Settings; // Organized as [cam0_rad_scale, cam0_rad_blend, cam0_drr_scale, cam0_drr_blend, cam1_rad_scale, cam1_rad_blend, cam1_drr_scale, cam1_drr_blend]

  // Default constructor
  FilterParticle();
  // Copy constructor
  FilterParticle(const FilterParticle& p);
  // Filter settings constructor
  FilterParticle(const std::vector<float>& filter_settings);
  // Random initialization constructor
  FilterParticle(float start_range_min, float start_range_max);
  // Assignment operator
  Particle& operator=(const Particle& p);
  void updateParticle(const Particle& pBest, const Particle& gBest, float omega);
};

extern std::ostream& operator<<(std::ostream& os, const FilterParticle& p);