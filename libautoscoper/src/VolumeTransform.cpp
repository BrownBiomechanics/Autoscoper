
#include "VolumeTransform.hpp"
#include <istream>
namespace xromm {

void VolumeTransform::setCurrentCurveSet(const int& idx)
{
  if (idx < 0 || idx >= numberOfCurveSets()) {
    std::cerr << "[WARNING] Failed to set curveSet " << idx << " is either below 0 or above " << numberOfCurveSets()
              << std::endl;
    return;
  }
  currentCurveSet = idx;
}

void VolumeTransform::addCurveSet()
{
  KeyCurve<float>* x_curve = new KeyCurve<float>(KeyCurve<float>::X_CURVE);
  KeyCurve<float>* y_curve = new KeyCurve<float>(KeyCurve<float>::Y_CURVE);
  KeyCurve<float>* z_curve = new KeyCurve<float>(KeyCurve<float>::Z_CURVE);
  KeyCurve<Quatf>* quat_curve = new KeyCurve<Quatf>(KeyCurve<Quatf>::QUAT_CURVE);

  x_curves.push_back(x_curve);
  y_curves.push_back(y_curve);
  z_curves.push_back(z_curve);
  quat_curves.push_back(quat_curve);

  currentCurveSet = numberOfCurveSets() - 1;
}

void VolumeTransform::setCurrentCurveSetToNext()
{
  // Increment the current curveSet or wrap it back to 0
  currentCurveSet++;
  if (currentCurveSet >= numberOfCurveSets()) {
    currentCurveSet = 0;
  }
}

void VolumeTransform::setCurrentCurveSetToPrevious()
{
  // Decrement the current curveSet or wrap it to the highest index
  currentCurveSet--;
  if (currentCurveSet < 0) {
    currentCurveSet = numberOfCurveSets() - 1;
  }
}

VolumeTransform::~VolumeTransform()
{
  x_curves.clear();
  y_curves.clear();
  z_curves.clear();
  quat_curves.clear();
}
} // namespace xromm
