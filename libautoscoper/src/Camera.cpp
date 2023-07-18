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
// ----------------------------------

/// \file Camera.cpp
/// \author Andy Loomis

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "Camera.hpp"

namespace xromm
{

  std::istream& safeGetline(std::istream& is, std::string& t)
  {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;)
    {
      int c = sb->sbumpc();
      switch (c)
      {
      case '\n':
        return is;
      case '\r':
        if (sb->sgetc() == '\n')
          sb->sbumpc();
        return is;
      case EOF:
        // Also handle the case when the last line has no line ending
        if (t.empty())
        {
          is.setstate(std::ios::eofbit);
        }
        return is;
      default:
        t += (char)c;
      }
    }
  }



Camera::Camera(const std::string& mayacam) : mayacam_(mayacam)
{
    // Load the mayacam.csv file into an array of doubles

    std::fstream file(mayacam.c_str(), std::ios::in);
    if (file.is_open() == false) {
        throw std::runtime_error("File not found: " + mayacam);
    }

    std::string csv_line;
  safeGetline(file, csv_line);
  file.close();
  if (csv_line.compare("image size") == 0)
  {
    loadMayaCam2(mayacam);
  }
  else
  {
    loadMayaCam1(mayacam);
  }

}


void Camera::loadMayaCam1(const std::string& mayacam)
  {
    std::fstream file(mayacam.c_str(), std::ios::in);
    double csv_vals[5][3];
    std::string csv_line, csv_val;
    for (int i = 0; i < 5 && safeGetline(file, csv_line); ++i) {
      std::istringstream csv_line_stream(csv_line);
      for (int j = 0; j < 3 && getline(csv_line_stream, csv_val, ','); ++j) {
        std::istringstream csv_val_stream(csv_val);
        if (!(csv_val_stream >> csv_vals[i][j])) {
          throw std::runtime_error("Invalid MayaCam file! Please check the mayacam 1.0 specification. https://autoscoper.readthedocs.io/en/latest/file-specifications/camera-calibration.html#mayacam-1-0");
        }
      }
    }
    file.close();

    // Line 1: Camera location in world space
    // Line 2: Rotations around the local x, y, and z axes of the camera. The
    //         order of rotation is z, y, then x.
    // Line 3: Position of the film plane relative to the camera. Given in the
    //         camera's local space. These values are calculated using the
    //         values in lines 4 and 5.
    // Line 4: u0, v0, Z
    // Line 5: scale, size_[0], size_[1]
    //
    // If the image width and height are set to 0 they are assumed to be 1024.
    //
    // The equations used to calculate the position and size of the film plane
    // are as follows:
    //
    // image_plane_x = scale*(size_[0]/2-u0)
    // image_plane_y = scale*(size_[1]/2-v0)
    // image_plane_z = -scale*Z
    //
    // image_plane_width = scale*size_[0]
    // image_plane_height = scale*size_[1]
    //
    // All units are in centimeters

    double* translation = csv_vals[0];
    double* rotation = csv_vals[1];
    double* image_plane_trans = csv_vals[2];

    double u0 = csv_vals[3][0] - 1; //Note: we should adjust here for the matlab offset of the old Mayacam
    double v0 = csv_vals[3][1] - 1;
    double z = csv_vals[3][2];

    // Default to 1024x1024
    size_[0] = csv_vals[4][1];
    if (size_[0] == 0) size_[0] = 1024;

    size_[1] = csv_vals[4][2];
    if (size_[1] == 0) size_[1] = 1024;

    // Calculate the cameras local coordinate frame
    double xyzypr[6] = { translation[0], translation[1], translation[2],
      rotation[2], rotation[1], rotation[0] };
    coord_frame_ = CoordFrame::from_xyzypr(xyzypr);

    // Calculate the viewport
    viewport_[0] = (2.0f*u0 - size_[0]) / z;
    viewport_[1] = (2.0f*v0 - size_[1]) / z;
    viewport_[2] = -2.0f*size_[0] / z;
    viewport_[3] = -2.0f*size_[1] / z;

    // Choose the scaling factor such that the image plane will be on the
    // other side of the origin from the camera. The values in the mayacam
    // file are discarded.
    double distance = sqrt(translation[0] * translation[0] +
      translation[1] * translation[1] +
      translation[2] * translation[2]);
    double scale = -1.5*distance / z;

    image_plane_trans[0] = scale*(size_[0] / 2.0 - u0);
    image_plane_trans[1] = scale*(size_[1] / 2.0 - v0);
    image_plane_trans[2] = scale*z;

    // Calculate the vertices at the corner of the image plane.
    double image_plane_center[3];
    coord_frame_.point_to_world_space(image_plane_trans, image_plane_center);

    double half_width = scale*size_[0] / 2.0;
    double half_height = scale*size_[1] / 2.0;

    double right[3] = { coord_frame_.rotation()[0],
      coord_frame_.rotation()[1],
      coord_frame_.rotation()[2] };
    double up[3] = { coord_frame_.rotation()[3],
      coord_frame_.rotation()[4],
      coord_frame_.rotation()[5] };

    image_plane_[0] = image_plane_center[0] - half_width*right[0] +
      half_height*up[0];
    image_plane_[1] = image_plane_center[1] - half_width*right[1] +
      half_height*up[1];
    image_plane_[2] = image_plane_center[2] - half_width*right[2] +
      half_height*up[2];

    image_plane_[3] = image_plane_center[0] - half_width*right[0] -
      half_height*up[0];
    image_plane_[4] = image_plane_center[1] - half_width*right[1] -
      half_height*up[1];
    image_plane_[5] = image_plane_center[2] - half_width*right[2] -
      half_height*up[2];

    image_plane_[6] = image_plane_center[0] + half_width*right[0] -
      half_height*up[0];
    image_plane_[7] = image_plane_center[1] + half_width*right[1] -
      half_height*up[1];
    image_plane_[8] = image_plane_center[2] + half_width*right[2] -
      half_height*up[2];

    image_plane_[9] = image_plane_center[0] + half_width*right[0] +
      half_height*up[0];
    image_plane_[10] = image_plane_center[1] + half_width*right[1] +
      half_height*up[1];
    image_plane_[11] = image_plane_center[2] + half_width*right[2] +
      half_height*up[2];
  }


  void Camera::loadMayaCam2(const std::string& mayacam)
  {
    double K[3][3];

    double rotation[3][3];
    double translation[3];


    std::fstream file(mayacam.c_str(), std::ios::in);
    double csv_vals[5][3];
    std::string csv_line, csv_val;
    for (int i = 0; i < 17 && safeGetline(file, csv_line); ++i) {
      std::istringstream csv_line_stream(csv_line);

      switch (i)
      {
        default:
          break;
        case 1: //size
          for (int j = 0; j < 2 && getline(csv_line_stream, csv_val, ','); ++j) {
            std::istringstream csv_val_stream(csv_val);
            if (!(csv_val_stream >> size_[j])) {
              throw std::runtime_error("Invalid MayaCam file! (size)");
            }
          }
          break;
        case 4: //K
        case 5:
        case 6:
          for (int j = 0; j < 3 && getline(csv_line_stream, csv_val, ','); ++j) {
            std::istringstream csv_val_stream(csv_val);
            if (!(csv_val_stream >> K[j][i - 4])) {
              throw std::runtime_error("Invalid MayaCam file! (K)");
            }
          }
          break;

        case 9: //R
        case 10:
        case 11:
          for (int j = 0; j < 3 && getline(csv_line_stream, csv_val, ','); ++j) {
            std::istringstream csv_val_stream(csv_val);
            if (!(csv_val_stream >> rotation[j][i - 9])) {
              throw std::runtime_error("Invalid MayaCam file! (R)");
            }
          }
          break;
        case 14: //t
        case 15:
        case 16:
          if (!(csv_line_stream >> translation[i - 14])) {
              throw std::runtime_error("Invalid MayaCam file! (t)");
            }

          break;

      }
    }
    file.close();

    //invert y - axis
    translation[0] = -translation[0];
    translation[2] = -translation[2];
    for (int i = 0; i < 3; i++)
    {
      rotation[i][0] = -rotation[i][0];
      rotation[i][2] = -rotation[i][2];
    }
    K[2][1] = (size_[1] - 1) - K[2][1];

    //invert rotation
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < i; j++)
      {

        double tmp = rotation[i][j];
        rotation[i][j] = rotation[j][i];
        rotation[j][i] = tmp;
      }
    }

    //invert translation
    {
      double translation_new[3];
      for (int i = 0; i < 3; i++)
        translation_new[i] = -(translation[0] * rotation[0][i] + translation[1] * rotation[1][i] + translation[2] * rotation[2][i]);

      for (int i = 0; i < 3; i++)
        translation[i] = translation_new[i];
    }

    for (int i = 0; i < 3; i++)
    {
      rotation[0][i] = -rotation[0][i];
      rotation[1][i] = -rotation[1][i];
    }

    coord_frame_ = CoordFrame(&rotation[0][0], translation);

    // Calculate the viewport
    viewport_[0] = -(2.0f*K[2][0] - size_[0]) / K[0][0];
    viewport_[1] = -(2.0f*K[2][1] - size_[1]) / K[1][1];
    viewport_[2] = 2.0f*size_[0] / K[0][0];
    viewport_[3] = 2.0f*size_[1] / K[1][1];


    // Choose the scaling factor such that the image plane will be on the
    // other side of the origin from the camera. The values in the mayacam
    // file are discarded.
    double z = - 0.5* (K[0][0] + K[1][1]);
    double distance = sqrt(translation[0] * translation[0] +
      translation[1] * translation[1] +
      translation[2] * translation[2]);
    double scale = -1.5*distance / z;

    double image_plane_trans[3];
    image_plane_trans[0] = scale*(size_[0] / 2.0 - K[2][0]);
    image_plane_trans[1] = scale*(size_[1] / 2.0 - K[2][1]);
    image_plane_trans[2] = scale*z;

    // Calculate the vertices at the corner of the image plane.
    double image_plane_center[3];
    coord_frame_.point_to_world_space(image_plane_trans, image_plane_center);

    double half_width = scale*size_[0] / 2.0;
    double half_height = scale*size_[1] / 2.0;

    double right[3] = { coord_frame_.rotation()[0],
      coord_frame_.rotation()[1],
      coord_frame_.rotation()[2] };
    double up[3] = { coord_frame_.rotation()[3],
      coord_frame_.rotation()[4],
      coord_frame_.rotation()[5] };

    image_plane_[0] = image_plane_center[0] - half_width*right[0] +
      half_height*up[0];
    image_plane_[1] = image_plane_center[1] - half_width*right[1] +
      half_height*up[1];
    image_plane_[2] = image_plane_center[2] - half_width*right[2] +
      half_height*up[2];

    image_plane_[3] = image_plane_center[0] - half_width*right[0] -
      half_height*up[0];
    image_plane_[4] = image_plane_center[1] - half_width*right[1] -
      half_height*up[1];
    image_plane_[5] = image_plane_center[2] - half_width*right[2] -
      half_height*up[2];

    image_plane_[6] = image_plane_center[0] + half_width*right[0] -
      half_height*up[0];
    image_plane_[7] = image_plane_center[1] + half_width*right[1] -
      half_height*up[1];
    image_plane_[8] = image_plane_center[2] + half_width*right[2] -
      half_height*up[2];

    image_plane_[9] = image_plane_center[0] + half_width*right[0] +
      half_height*up[0];
    image_plane_[10] = image_plane_center[1] + half_width*right[1] +
      half_height*up[1];
    image_plane_[11] = image_plane_center[2] + half_width*right[2] +
      half_height*up[2];
  }

} // namespace XROMM
