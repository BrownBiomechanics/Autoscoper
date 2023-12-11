#ifndef XROMM_EOS_RAY_CASTER_HPP
#define XROMM_EOS_RAY_CASTER_HPP

#include <string>

#include "OpenCL.hpp"
#include "VolumeDescription.hpp"
#include "CoordFrame.hpp"

namespace xromm {
  namespace gpu {
    class EOSRayCaster {
    public:
      EOSRayCaster();
      ~EOSRayCaster();

      void setVolume(VolumeDescription& volume);
      void setWorldToModelMatrix(CoordFrame& worldToModelMatrix);
      void setGeometry(const float& sid, const float& sdd, const float& lambda, const float& lambdaZ, const float& cutoff, const float& intensity, bool isLateral, const unsigned int& R, const unsigned int& C);
      void render(const Buffer* buffer, unsigned int width, unsigned int height);

      void calculateViewport();

      double* viewport() {
        return viewport_;
      }

      float getSourceToIsocenterDistance() const {
        return sid_;
      }
      void setSourceToIsocenterDistance(const float& sid) {
        sid_ = sid;
      }
      float getSourceToDetectorDistance() const {
        return sdd_;
      }
      void setSourceToDetectorDistance(const float& sdd) {
        sdd_ = sdd;
      }
      float getLambda() const {
        return lambda_;
      }
      void setLambda(const float& lambda) {
        lambda_ = lambda;
      }
      float getLambdaZ() const {
        return lambdaZ_;
      }
      void setLambdaZ(const float& lambdaZ) {
        lambdaZ_ = lambdaZ;
      }
      float getCutoff() const {
        return cutoff_;
      }
      float getMinCutoff() const {
        return volumeDescription_->minValue();
      }
      float getMaxCutoff() const {
        return volumeDescription_->maxValue();
      }
      void setCutoff(const float& cutoff) {
        cutoff_ = cutoff;
      }
      const std::string& getName() const {
        return name_;
      }
      void setName(const std::string& name) {
        name_ = name;
      }
      void setVisible(const bool& visible) {
        visible_ = visible;
      }


    private:
      CoordFrame* worldToModelMatrix_;
      double worldToModelArray_[16] = { 0.0 };
      Buffer* worldToModelBuffer_;
      Buffer* viewportBuffer_;
      std::string name_;
      VolumeDescription* volumeDescription_;
      float sid_;
      float sdd_;
      float lambda_;
      float lambdaZ_;
      float cutoff_;
      float intensity_;
      unsigned int isLateral_;
      bool visible_;
      unsigned int z0_;
      unsigned int R_;
      unsigned int C_;
      double viewport_[4] = {0.0};
    };

} } // namespace xromm::gpu
#endif // XROMM_EOS_RAY_CASTER_HPP