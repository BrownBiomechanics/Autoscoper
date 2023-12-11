#include "EOSRayCaster.hpp"
#include <sstream>

namespace xromm {
  namespace gpu {
#include "gpu/opencl/kernel/EosRayCaster.cl.h"

#define BX 16
#define BY 16

    static Program eos_ray_caster_program_;
    static int num_casters = 0;

    EOSRayCaster::EOSRayCaster() : volumeDescription_(0),
                                   name_(""),
                                   sid_(0.f),
                                   sdd_(0.f),
                                   lambda_(0.f),
                                   lambdaZ_(0.f),
                                   cutoff_(0.f),
                                   intensity_(0.f),
                                   isLateral_(0),
                                   z0_(0),
                                   R_(0),
                                   C_(0) {
      std::stringstream name_stream;
      name_stream << "EOS DRR Renderer" << (++num_casters);
      name_ = name_stream.str();
      visible_ = true;
      worldToModelArray_[0] = 1.f;
      worldToModelArray_[5] = 1.f;
      worldToModelArray_[10] = 1.f;
      worldToModelArray_[15] = 1.f;
      worldToModelBuffer_ = new Buffer(16 * sizeof(float), CL_MEM_READ_ONLY);
      worldToModelBuffer_->read(worldToModelArray_);
      worldToModelMatrix_ = new CoordFrame();
      worldToModelMatrix_->from_matrix(&(worldToModelArray_[0]));

      viewportBuffer_ = new Buffer(4 * sizeof(float), CL_MEM_READ_ONLY);
    }

    EOSRayCaster::~EOSRayCaster() {
      if (worldToModelBuffer_) delete worldToModelBuffer_;
      if (viewportBuffer_) delete viewportBuffer_;
    }

    void EOSRayCaster::setVolume(VolumeDescription& volume) {
      volumeDescription_ = &volume;
    }

    void EOSRayCaster::setWorldToModelMatrix(CoordFrame& worldToModelMatrix) {
      worldToModelMatrix_ = &worldToModelMatrix;
      worldToModelMatrix_->to_matrix(&(worldToModelArray_[0])); // Column-major
      worldToModelBuffer_->read(worldToModelArray_);
    }

    void EOSRayCaster::setGeometry(const float& sid, const float& sdd, const float& lambda, const float& lambdaZ, const float& cutoff, const float& intensity, bool isLateral, const unsigned int& R, const unsigned int& C) {
      sid_ = sid;
      sdd_ = sdd;
      lambda_ = lambda;
      lambdaZ_ = lambdaZ;
      cutoff_ = cutoff;
      intensity_ = intensity;
      isLateral_ = (unsigned int)isLateral;
      R_ = R;
      C_ = C;
    }

    void EOSRayCaster::render(const Buffer* buffer, unsigned int width, unsigned int height) {
      if (!volumeDescription_) {
        std::cerr << "EOSRayCaster: WARNING: No volume loaded." << std::endl;
        return;
      }

      if (!visible_) {
        buffer->fill((char)0x00);
        return;
      }

      if (!viewportBuffer_) {
        std::cerr << "EOSRayCaster: WARNING: No viewport loaded." << std::endl;
        return;
      }

      Kernel* kernel = eos_ray_caster_program_.compile(EosRayCaster_cl, "eos_project_drr");

      kernel->block2d(BX, BY);
      kernel->grid2d((height + BX - 1) / BX, (width + BY - 1) / BY);

      // add offset width,height

      kernel->addBufferArg(buffer);
      kernel->addImageArg(volumeDescription_->image());
      kernel->addBufferArg(worldToModelBuffer_);
      kernel->addBufferArg(viewportBuffer_);
      kernel->addArg(z0_);
      kernel->addArg(lambda_);
      kernel->addArg(lambdaZ_);
      kernel->addArg(sid_);
      kernel->addArg(sdd_);
      kernel->addArg(C_);
      kernel->addArg(R_);
      kernel->addArg(isLateral_);
      kernel->addArg(cutoff_);
      kernel->addArg(intensity_);
      kernel->addArg(width);
      kernel->addArg(height);

      kernel->launch();

      delete kernel;
    }

    void EOSRayCaster::calculateViewport() {
      if (!volumeDescription_) {
        std::cerr << "EOSRayCaster: WARNING: No volume loaded." << std::endl;
        return;
      }
      if (!worldToModelMatrix_) {
        std::cerr << "EOSRayCaster: WARNING: No world to model matrix." << std::endl;
        return;
      }

      double corners[24] = { 0,0,-1,0,0,0, 0,1,-1,0,1,0, 1,0,-1,1,0,0,1,1,-1,1,1,0 };
      double min_max[4] = { C_, R_, 0, 0 }; // min_x, min_y, max_x, max_y

      for (int j = 0; j < 8; j++) {
        // Calculate the location of each corner in object space
        corners[3 * j + 0] = (corners[3 * j + 0] - volumeDescription_->invTrans()[0]) / volumeDescription_->invScale()[0];
        corners[3 * j + 1] = (corners[3 * j + 1] - volumeDescription_->invTrans()[1]) / volumeDescription_->invScale()[1];
        corners[3 * j + 2] = (corners[3 * j + 2] - volumeDescription_->invTrans()[2]) / volumeDescription_->invScale()[2];

        // Move the corner into world space
        double corner_world[3];
        worldToModelMatrix_->inverse().point_to_world_space(&corners[3*j], corner_world);

        // Project the corner onto the radiograph plane
        double corner_image[2];
        if (!isLateral_)
          corner_image[0] = (C_ / 2) - (sid_ * corner_world[0]) / (lambda_ * (sid_ + corner_world[1]));
        else
          corner_image[0] = (C_ / 2) - (sid_ * corner_world[1]) / (lambda_ * (sid_ + corner_world[2]));
        corner_image[1] = (z0_ - corner_world[2]) / lambdaZ_;

        // Make sure the corner is within the radiograph
        if (corner_image[0] < 0) corner_image[0] = 0;
        if (corner_image[0] > C_) corner_image[0] = C_;
        if (corner_image[1] < 0) corner_image[1] = 0;
        if (corner_image[1] > R_) corner_image[1] = R_;

        // Update the bounding box
        if (corner_image[0] < min_max[0]) min_max[0] = corner_image[0];
        if (corner_image[0] > min_max[2]) min_max[2] = corner_image[0];
        if (corner_image[1] < min_max[1]) min_max[1] = corner_image[1];
        if (corner_image[1] > min_max[3]) min_max[3] = corner_image[1];
      }

      // Calculate the viewport
      //viewport_[0] = (int)floor(min_max[0]); // x
      //viewport_[1] = (int)floor(min_max[1]); // y
      //viewport_[2] = (int)ceil(min_max[2]) - viewport_[0]; // width
      //viewport_[3] = (int)ceil(min_max[3]) - viewport_[1]; // height

      viewport_[0] = 0;
      viewport_[1] = 0;
      viewport_[2] = C_;
      viewport_[3] = R_;

      viewportBuffer_->read(viewport_);

    }
} } // namespace xromm::gpu