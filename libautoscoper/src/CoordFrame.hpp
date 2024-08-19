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

/// \file CoordFrame.hpp
/// \author Andy Loomis, Benjamin Knorlein

#ifndef XROMM_COORD_FRAME_HPP
#define XROMM_COORD_FRAME_HPP

#include <string>
#include <Vector.hpp>

namespace xromm {

// Class representing a coordinate transformation
// Based on the G3D Coordinate Frame class
// All rotations are based in degrees

class CoordFrame
{
public:
  CoordFrame();

  CoordFrame(const double* rotation, const double* translation);

  CoordFrame(const CoordFrame& xcframe);

  ~CoordFrame() {}

  static CoordFrame from_xyzypr(const double* xyzypr);

  static CoordFrame from_xyzquat(const double* xyzijk);

  static CoordFrame from_xyzAxis_angle(const double* xyzijk);

  void to_xyzypr(double* xyzypr) const;

  static CoordFrame from_matrix(const double* m);

  void to_matrix(double* m) const;

  void to_matrix_row_order(double* m) const;

  void orient(const double* rotation, const double* translation);

  void translate(const double* v);

  void set_translation(const Vec3f& t);

  void rotate(const double* v, double angle);

  CoordFrame inverse() const;

  void point_to_world_space(const double* p, double* q) const;

  void vector_to_world_space(const double* p, double* q) const;

  Vec3f rotate_vector(const Vec3f& p);

  Vec3f translate_vector(const Vec3f& p);

  Vec3f transform_vector(const Vec3f& p);

  CoordFrame linear_extrap(const CoordFrame& x2) const;

  CoordFrame operator*(const CoordFrame& xcframe) const;

  CoordFrame& operator=(const CoordFrame& xcframe);

  // Accessors

  inline double* translation() { return translation_; }

  inline const double* translation() const { return translation_; }

  inline double* rotation() { return rotation_; }

  inline const double* rotation() const { return rotation_; }

  // Formatting

  std::string to_string() const;

  void from_string(std::string str);

private:
  double rotation_[9];

  double translation_[3];

  void rotateQuat(double x, double y, double z);
};

std::ostream& operator<<(std::ostream& os, const CoordFrame& frame);
} // namespace xromm

#endif // XROMM_COORD_FRAME_HPP
