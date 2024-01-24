
#include "VolumeTransform.hpp"

namespace xromm {
VolumeTransform::VolumeTransform()
{
  x_curve.type = KeyCurve<float>::X_CURVE;
  y_curve.type = KeyCurve<float>::Y_CURVE;
  z_curve.type = KeyCurve<float>::Z_CURVE;
  quat_curve.type = KeyCurve<Quatf>::QUAT_CURVE;
}

VolumeTransform::~VolumeTransform() {}
} // namespace xromm