    
#include "VolumeTransform.hpp"

namespace xromm
{
	VolumeTransform::VolumeTransform(){
		x_curve.type = KeyCurve::X_CURVE;
		y_curve.type = KeyCurve::Y_CURVE;
		z_curve.type = KeyCurve::Z_CURVE;
		yaw_curve.type = KeyCurve::YAW_CURVE;
		pitch_curve.type = KeyCurve::PITCH_CURVE;
		roll_curve.type = KeyCurve::ROLL_CURVE;
	}

	VolumeTransform::~VolumeTransform(){

	}
}