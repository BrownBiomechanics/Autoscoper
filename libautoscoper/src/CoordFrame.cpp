// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

/// \file CoordFrame.cpp
/// \author Andy Loomis, Benjamin Knorlein

#include "CoordFrame.hpp"

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define sqr(a) ((a) * (a))

namespace xromm
{

CoordFrame::CoordFrame()
{
    rotation_[0] = 1.0;
    rotation_[1] = 0.0;
    rotation_[2] = 0.0;

    rotation_[3] = 0.0;
    rotation_[4] = 1.0;
    rotation_[5] = 0.0;

    rotation_[6] = 0.0;
    rotation_[7] = 0.0;
    rotation_[8] = 1.0;

    translation_[0] = 0.0;
    translation_[1] = 0.0;
    translation_[2] = 0.0;
}

CoordFrame::CoordFrame(const double* rotation,
                         const double* translation)
{
    this->orient(rotation, translation);
}

CoordFrame::CoordFrame(const CoordFrame& xcframe)
{
    rotation_[0] = xcframe.rotation_[0];
    rotation_[1] = xcframe.rotation_[1];
    rotation_[2] = xcframe.rotation_[2];

    rotation_[3] = xcframe.rotation_[3];
    rotation_[4] = xcframe.rotation_[4];
    rotation_[5] = xcframe.rotation_[5];

    rotation_[6] = xcframe.rotation_[6];
    rotation_[7] = xcframe.rotation_[7];
    rotation_[8] = xcframe.rotation_[8];

    translation_[0] = xcframe.translation_[0];
    translation_[1] = xcframe.translation_[1];
    translation_[2] = xcframe.translation_[2];
}

CoordFrame CoordFrame::from_xyzypr(const double* xyzypr)
{
    CoordFrame xcframe;
    xcframe.rotate(xcframe.rotation_+6, xyzypr[3]);
    xcframe.rotate(xcframe.rotation_+3, xyzypr[4]);
    xcframe.rotate(xcframe.rotation_+0, xyzypr[5]);
    xcframe.translate(xyzypr); //Only uses xyz
    return xcframe;
}

CoordFrame CoordFrame::from_xyzquat(const double* xyzijk)
{
  CoordFrame xcframe;
  xcframe.rotateQuat(xyzijk[3], xyzijk[4], xyzijk[5]);
  xcframe.translate(xyzijk); //Only uses xyz
  return xcframe;
}

void CoordFrame::rotateQuat(double x, double y, double z){

  double w = sqrt(1 - (sqr(x) + sqr(y) + sqr(z)));
  double x_a, y_a, z_a, angle;
  if (w != 1){
    angle = 2 * acos(w);
    x_a = x / sqrt(1 - sqr(w));
    y_a = y / sqrt(1 - sqr(w));
    z_a = z / sqrt(1 - sqr(w));
  }
  else  {
    x_a = y_a = z_a, angle = 0;
  }

  double c = cos(angle);
  double s = sin(angle);
  double t = 1 - c;

  double R[9] = {

    t * sqr(x_a) + c,
    t * x_a * y_a - z_a * s,
    t * x_a * z_a + y_a * s,
    t * x_a * y_a + z_a * s,
    t * sqr(y_a) + c,
    t * y_a * z_a - x_a * s,
    t * x_a * z_a - y_a * s,
    t * y_a * z_a + x_a * s,
    t * sqr(z_a) + c

  };

  double* M = rotation_;

  double temp[9] = { M[0] * R[0] + M[1] * R[3] + M[2] * R[6],
    M[0] * R[1] + M[1] * R[4] + M[2] * R[7],
    M[0] * R[2] + M[1] * R[5] + M[2] * R[8],
    M[3] * R[0] + M[4] * R[3] + M[5] * R[6],
    M[3] * R[1] + M[4] * R[4] + M[5] * R[7],
    M[3] * R[2] + M[4] * R[5] + M[5] * R[8],
    M[6] * R[0] + M[7] * R[3] + M[8] * R[6],
    M[6] * R[1] + M[7] * R[4] + M[8] * R[7],
    M[6] * R[2] + M[7] * R[5] + M[8] * R[8] };

  rotation_[0] = temp[0];
  rotation_[1] = temp[1];
  rotation_[2] = temp[2];
  rotation_[3] = temp[3];
  rotation_[4] = temp[4];
  rotation_[5] = temp[5];
  rotation_[6] = temp[6];
  rotation_[7] = temp[7];
  rotation_[8] = temp[8];
}

CoordFrame CoordFrame::from_xyzAxis_angle(const double* xyzijk)
{
  CoordFrame xcframe;

  double angle = sqrt(sqr(xyzijk[3]) + sqr(xyzijk[4]) + sqr(xyzijk[5]));

  if (angle != 0)
    xcframe.rotate(&xyzijk[3], angle);

  xcframe.translate(xyzijk);

  return xcframe;
}

void CoordFrame::to_xyzypr(double* xyzypr) const
{
    xyzypr[0] = translation_[0];
    xyzypr[1] = translation_[1];
    xyzypr[2] = translation_[2];
    xyzypr[3] = 180.0*atan2(rotation_[1], rotation_[0])/M_PI;
    xyzypr[4] = 180.0*atan2(-rotation_[2],
                            sqrt(rotation_[5]*rotation_[5]+
                            rotation_[8]*rotation_[8]))/M_PI;
    xyzypr[5] = 180*atan2(rotation_[5], rotation_[8])/M_PI;
}

CoordFrame CoordFrame::from_matrix(const double* m)
{
    double rotation[9] = {m[0],  m[1],  m[2],
                          m[4],  m[5],  m[6],
                          m[8],  m[9],  m[10]};
    double translation[3] = {m[12], m[13], m[14]};

    return CoordFrame(rotation, translation);
}

void CoordFrame::to_matrix(double* m) const
{
    m[0]  = rotation_[0];
    m[1]  = rotation_[1];
    m[2]  = rotation_[2];
    m[3]  = 0.0;
    m[4]  = rotation_[3];
    m[5]  = rotation_[4];
    m[6]  = rotation_[5];
    m[7]  = 0.0;
    m[8]  = rotation_[6];
    m[9]  = rotation_[7];
    m[10] = rotation_[8];
    m[11] = 0.0;
    m[12] = translation_[0];
    m[13] = translation_[1];
    m[14] = translation_[2];
    m[15] = 1.0;
}

void CoordFrame::to_matrix_row_order(double* m) const
{
    m[0]  = rotation_[0];
    m[1]  = rotation_[3];
    m[2]  = rotation_[6];
    m[3]  = translation_[0];
    m[4]  = rotation_[1];
    m[5]  = rotation_[4];
    m[6]  = rotation_[7];
    m[7]  = translation_[1];
    m[8]  = rotation_[2];
    m[9]  = rotation_[5];
    m[10] = rotation_[8];
    m[11] = translation_[2];
    m[12] = 0.0;
    m[13] = 0.0;
    m[14] = 0.0;
    m[15] = 1.0;
}


void CoordFrame::orient(const double* rotation,
                         const double* translation)
{
    rotation_[0] = rotation[0];
    rotation_[1] = rotation[1];
    rotation_[2] = rotation[2];

    rotation_[3] = rotation[3];
    rotation_[4] = rotation[4];
    rotation_[5] = rotation[5];

    rotation_[6] = rotation[6];
    rotation_[7] = rotation[7];
    rotation_[8] = rotation[8];

    translation_[0] = translation[0];
    translation_[1] = translation[1];
    translation_[2] = translation[2];
}

void CoordFrame::rotate(const double* caxis, double angle)
{
    double axis[3] = {caxis[0], caxis[1], caxis[2]};

    double c = cos(M_PI*angle/180.0);
    double s = sin(M_PI*angle/180.0);

    // Normalize the axis of rotation
    double mag = sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
    if (mag == 0) {
        return;
    }
    axis[0] /= mag;
    axis[1] /= mag;
    axis[2] /= mag;

    double xx = axis[0]*axis[0];
    double yy = axis[1]*axis[1];
    double zz = axis[2]*axis[2];
    double xy = axis[0]*axis[1];
    double xz = axis[0]*axis[2];
    double yz = axis[1]*axis[2];
    double sx = s*axis[0];
    double sy = s*axis[1];
    double sz = s*axis[2];
    double oc = 1.0-c;

    // Calculate the rotation matrix
    double R[9] = {(oc*xx)+c,  (oc*xy)+sz, (oc*xz)-sy,
                   (oc*xy)-sz, (oc*yy)+c,  (oc*yz)+sx,
                   (oc*xz)+sy, (oc*yz)-sx, (oc*zz)+c};

    double* M = rotation_;

    double temp[9] = {M[0]*R[0]+M[1]*R[3]+M[2]*R[6],
                      M[0]*R[1]+M[1]*R[4]+M[2]*R[7],
                      M[0]*R[2]+M[1]*R[5]+M[2]*R[8],
                      M[3]*R[0]+M[4]*R[3]+M[5]*R[6],
                      M[3]*R[1]+M[4]*R[4]+M[5]*R[7],
                      M[3]*R[2]+M[4]*R[5]+M[5]*R[8],
                      M[6]*R[0]+M[7]*R[3]+M[8]*R[6],
                      M[6]*R[1]+M[7]*R[4]+M[8]*R[7],
                      M[6]*R[2]+M[7]*R[5]+M[8]*R[8]};

    rotation_[0] = temp[0];
    rotation_[1] = temp[1];
    rotation_[2] = temp[2];
    rotation_[3] = temp[3];
    rotation_[4] = temp[4];
    rotation_[5] = temp[5];
    rotation_[6] = temp[6];
    rotation_[7] = temp[7];
    rotation_[8] = temp[8];
}

void CoordFrame::translate(const double* v)
{
    translation_[0] += v[0];
    translation_[1] += v[1];
    translation_[2] += v[2];
}

CoordFrame CoordFrame::inverse() const
{
    double rotation[9] = {rotation_[0], rotation_[3], rotation_[6],
                          rotation_[1], rotation_[4], rotation_[7],
                          rotation_[2], rotation_[5], rotation_[8]};
    double translation[3] = {-(rotation[0]*translation_[0]+
                               rotation[3]*translation_[1]+
                               rotation[6]*translation_[2]),
                             -(rotation[1]*translation_[0]+
                               rotation[4]*translation_[1]+
                               rotation[7]*translation_[2]),
                             -(rotation[2]*translation_[0]+
                               rotation[5]*translation_[1]+
                               rotation[8]*translation_[2])};

    return CoordFrame(rotation, translation);
}

void CoordFrame::point_to_world_space(const double* p, double* q) const
{
    q[0] = rotation_[0]*p[0]+
           rotation_[3]*p[1]+
           rotation_[6]*p[2]+
           translation_[0];
    q[1] = rotation_[1]*p[0]+
           rotation_[4]*p[1]+
           rotation_[7]*p[2]+
           translation_[1];
    q[2] = rotation_[2]*p[0]+
           rotation_[5]*p[1]+
           rotation_[8]*p[2]+
           translation_[2];
}

void CoordFrame::vector_to_world_space(const double* p, double* q) const
{
    q[0] = rotation_[0]*p[0]+
           rotation_[3]*p[1]+
           rotation_[6]*p[2];
    q[1] = rotation_[1]*p[0]+
           rotation_[4]*p[1]+
           rotation_[7]*p[2];
    q[2] = rotation_[2]*p[0]+
           rotation_[5]*p[1]+
           rotation_[8]*p[2];
}

CoordFrame CoordFrame::linear_extrap(const CoordFrame& x) const
{
    double t = 2.0;

    double trans[3] = { t*(x.translation_[0]-translation_[0]),
                        t*(x.translation_[1]-translation_[1]),
                        t*(x.translation_[2]-translation_[2]) };

    double A[9] = {
        x.rotation_[0]*rotation_[0]+x.rotation_[3]*rotation_[3]+
        x.rotation_[6]*rotation_[6],
        x.rotation_[1]*rotation_[0]+x.rotation_[4]*rotation_[3]+
        x.rotation_[7]*rotation_[6],
        x.rotation_[2]*rotation_[0]+x.rotation_[5]*rotation_[3]+
        x.rotation_[8]*rotation_[6],
        x.rotation_[0]*rotation_[1]+x.rotation_[3]*rotation_[4]+
        x.rotation_[6]*rotation_[7],
        x.rotation_[1]*rotation_[1]+x.rotation_[4]*rotation_[4]+
        x.rotation_[7]*rotation_[7],
        x.rotation_[2]*rotation_[1]+x.rotation_[5]*rotation_[4]+
        x.rotation_[8]*rotation_[7],
        x.rotation_[0]*rotation_[2]+x.rotation_[3]*rotation_[5]+
        x.rotation_[6]*rotation_[8],
        x.rotation_[1]*rotation_[2]+x.rotation_[4]*rotation_[5]+
        x.rotation_[7]*rotation_[8],
        x.rotation_[2]*rotation_[2]+x.rotation_[5]*rotation_[5]+
        x.rotation_[8]*rotation_[8] };

    double axis[3];
    double angleRadians;

    double trace = A[0]+A[4]+A[8];
    angleRadians = acos((trace-1.0)/2.0);
    if (angleRadians > 0.0) {
        if (angleRadians < M_PI) {
            axis[0] = A[5]-A[7];
            axis[1] = A[6]-A[2];
            axis[2] = A[1]-A[3];
        }
        else {
            std::cerr << "TODO: Unable to determine angle." << std::endl;
        }
    }
    // Angle is zero, use any axis
    else {
        axis[0] = 1.0;
        axis[1] = 0.0;
        axis[2] = 0.0;
    }

    CoordFrame x3 = *this;
    x3.rotate(axis,t*180.0*angleRadians/M_PI);
    x3.translate(trans);
    return x3;
}

CoordFrame CoordFrame::operator*(const CoordFrame& xcframe) const
{
    double rotation[9], translation[3];

    rotation[0] = rotation_[0]*xcframe.rotation_[0]+
                  rotation_[3]*xcframe.rotation_[1]+
                  rotation_[6]*xcframe.rotation_[2];
    rotation[1] = rotation_[1]*xcframe.rotation_[0]+
                  rotation_[4]*xcframe.rotation_[1]+
                  rotation_[7]*xcframe.rotation_[2];
    rotation[2] = rotation_[2]*xcframe.rotation_[0]+
                  rotation_[5]*xcframe.rotation_[1]+
                  rotation_[8]*xcframe.rotation_[2];

    rotation[3] = rotation_[0]*xcframe.rotation_[3]+
                  rotation_[3]*xcframe.rotation_[4]+
                  rotation_[6]*xcframe.rotation_[5];
    rotation[4] = rotation_[1]*xcframe.rotation_[3]+
                  rotation_[4]*xcframe.rotation_[4]+
                  rotation_[7]*xcframe.rotation_[5];
    rotation[5] = rotation_[2]*xcframe.rotation_[3]+
                  rotation_[5]*xcframe.rotation_[4]+
                  rotation_[8]*xcframe.rotation_[5];

    rotation[6] = rotation_[0]*xcframe.rotation_[6]+
                  rotation_[3]*xcframe.rotation_[7]+
                  rotation_[6]*xcframe.rotation_[8];
    rotation[7] = rotation_[1]*xcframe.rotation_[6]+
                  rotation_[4]*xcframe.rotation_[7]+
                  rotation_[7]*xcframe.rotation_[8];
    rotation[8] = rotation_[2]*xcframe.rotation_[6]+
                  rotation_[5]*xcframe.rotation_[7]+
                  rotation_[8]*xcframe.rotation_[8];

    translation[0] = rotation_[0]*xcframe.translation_[0]+
                     rotation_[3]*xcframe.translation_[1]+
                     rotation_[6]*xcframe.translation_[2]+
                     translation_[0];
    translation[1] = rotation_[1]*xcframe.translation_[0]+
                     rotation_[4]*xcframe.translation_[1]+
                     rotation_[7]*xcframe.translation_[2]+
                     translation_[1];
    translation[2] = rotation_[2]*xcframe.translation_[0]+
                     rotation_[5]*xcframe.translation_[1]+
                     rotation_[8]*xcframe.translation_[2]+
                     translation_[2];

    return CoordFrame(rotation, translation);
}

CoordFrame& CoordFrame::operator=(const CoordFrame& xcframe)
{
    rotation_[0] = xcframe.rotation_[0];
    rotation_[1] = xcframe.rotation_[1];
    rotation_[2] = xcframe.rotation_[2];

    rotation_[3] = xcframe.rotation_[3];
    rotation_[4] = xcframe.rotation_[4];
    rotation_[5] = xcframe.rotation_[5];

    rotation_[6] = xcframe.rotation_[6];
    rotation_[7] = xcframe.rotation_[7];
    rotation_[8] = xcframe.rotation_[8];

    translation_[0] = xcframe.translation_[0];
    translation_[1] = xcframe.translation_[1];
    translation_[2] = xcframe.translation_[2];

    return *this;
}

std::string CoordFrame::to_string() const
{
    std::stringstream ss;
    ss << rotation_[0] << ", "
       << rotation_[1] << ", "
       << rotation_[2] << ", "
       << 0.0 << ", "
       << rotation_[3] << ", "
       << rotation_[4] << ", "
       << rotation_[5] << ", "
       << 0.0 << ", "
       << rotation_[6] << ", "
       << rotation_[7] << ", "
       << rotation_[8] << ", "
       << 0.0 << ", "
       << translation_[0] << ", "
       << translation_[1] << ", "
       << translation_[2] << ", "
       << 1.0;

    return ss.str();
}

void CoordFrame::from_string(std::string str)
{
  std::vector< double > vd;
    double d = 0.0;
  std::size_t pos = 0;
    while (pos < str.size ())
        if ((pos = str.find_first_of (',',pos)) != std::string::npos)
            str[pos] = ' ';

  std::stringstream ss(str);

  while (ss >> d)
        vd.push_back (d);

  rotation_[0] = vd[0];
  rotation_[2] = vd[1];
  rotation_[1] = vd[2];

  rotation_[3] = vd[4];
  rotation_[4] = vd[5];
  rotation_[5]= vd[6];

  rotation_[6] = vd[8];
  rotation_[7] = vd[9];
  rotation_[8]= vd[10];

  translation_[0]= vd[12];
  translation_[1] = vd[13];
  translation_[2] = vd[14];
}

std::ostream& operator<<(std::ostream& os, const CoordFrame& frame)
{
  os << frame.rotation()[0] << " , " << frame.rotation()[1] << " , " << frame.rotation()[2] << frame.translation()[0] << std::endl;
  os << frame.rotation()[3] << " , " << frame.rotation()[4] << " , " << frame.rotation()[5] << frame.translation()[1] << std::endl;
  os << frame.rotation()[6] << " , " << frame.rotation()[7] << " , " << frame.rotation()[8] << frame.translation()[2] << std::endl;
  return os;
}

} // namespace XROMM
