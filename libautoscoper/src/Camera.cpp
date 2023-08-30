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

#include <algorithm>
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

  std::string mayaCamReadingError(const std::string& version, int line, const std::string& filename, const std::string& message)
  {
    return std::string("Invalid MayaCam ") + version + ".0 file. " + message + "\n"
          "See line " + std::to_string(line) + " in " + filename + ".\n"
          "\n"
          "Please check the MayaCam " + version + ".0 specification.\n"
          "See https://autoscoper.readthedocs.io/en/latest/file-specifications/camera-calibration.html#mayacam-" + version + "-0";
  }

  std::string vtkCamReadingError(const std::string& version, int line, const std::string& filename, const std::string& message) {
    return std::string("Invalid VTKCam ") + version + ".0 file. " + message + "\n"
      "See line " + std::to_string(line) + " in " + filename + ".\n"
      "\n"
      "Please check the VTKCam " + version + ".0 specification.\n"
      "See https://autoscoper.readthedocs.io/en/latest/file-specifications/camera-calibration.html#vtkcam-" + version + "-0";
  }

  bool parseArray(std::string& value, double* a, int n) {
    value.erase(std::remove(value.begin(), value.end(), '['), value.end());
    value.erase(std::remove(value.begin(), value.end(), ']'), value.end());
    value.erase(std::remove(value.begin(), value.end(), ','), value.end());
    std::istringstream value_stream(value);
    for (int i = 0; i < n; ++i) {
      if (!(value_stream >> a[i])) {
        return false;
      }
    }
    return true;
  }

Camera::Camera(const std::string& mayacam) : mayacam_(mayacam)
{
    // Check the file extension
    std::string::size_type ext_pos = mayacam_.find_last_of('.');
    if (ext_pos == std::string::npos) {
        throw std::runtime_error("Invalid MayaCam file");
    }
    std::string ext = mayacam_.substr(ext_pos + 1);
    // if its a yaml file load it as a vtk camera
    if (ext.compare("yaml") == 0) {
        loadVTKCamera(mayacam_);
    }
    else {
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
}

void Camera::loadMayaCam1(const std::string& mayacam)
  {
    std::cout << "Reading MayaCam 1.0 file: " << mayacam << std::endl;

    std::fstream file(mayacam.c_str(), std::ios::in);
    double csv_vals[5][3] = {0.};
    std::string csv_line, csv_val;
    int line_count = 0;

    for (int i = 0; i < 5 && safeGetline(file, csv_line); ++i) {
      int read_count = 0;
      std::istringstream csv_line_stream(csv_line);
      for (int j = 0; j < 3 && std::getline(csv_line_stream, csv_val, ','); ++j) {
        std::istringstream csv_val_stream(csv_val);
        if (!(csv_val_stream >> csv_vals[i][j])) {
          break;
        }
        ++read_count;
      }
      if (read_count != 3) {
        throw std::runtime_error(
              mayaCamReadingError("1", /* line= */ i + 1, mayacam, "There was an error reading values."));
      }
      ++line_count;
    }
    file.close();

    if (line_count != 5) {
      throw std::runtime_error(
            mayaCamReadingError("1", /* line= */ 1, mayacam, "There was an error reading values."));
    }

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
    if (z < 0) {
      calculateViewport(u0, v0, -z, -z);
    } else {
      calculateViewport(u0, v0, z, z);
    }

    // Calculate the Image Plane
    calculateImagePlane(u0, v0, z);
  }

  void Camera::loadMayaCam2(const std::string& mayacam)
  {
    std::cout << "Reading MayaCam 2.0 file: " << mayacam << std::endl;

    // camera matrix
    double K[3][3] = {0.};

    // rotation
    double rotation[3][3] = {0.};

    // translation
    double translation[3] = {0.};
    int translation_read_count = 0;


    std::fstream file(mayacam.c_str(), std::ios::in);
    std::string csv_line, csv_val;
    for (int i = 0; i < 17 && safeGetline(file, csv_line); ++i) {
      std::istringstream csv_line_stream(csv_line);

      switch (i)
      {
        default:
          break;
        case 1: //size
        {
          int read_count = 0;
          for (int j = 0; j < 2 && std::getline(csv_line_stream, csv_val, ','); ++j) {
            std::istringstream csv_val_stream(csv_val);
            if (!(csv_val_stream >> size_[j])) {
              break;
            }
            ++read_count;
          }
          if (read_count != 2) {
            throw std::runtime_error(
                  mayaCamReadingError("2", /* line= */ i + 1, mayacam, "There was an error reading 'image size' values."));
          }
          break;
        }
        case 4: //K
        case 5:
        case 6:
        {
          int read_count = 0;
          for (int j = 0; j < 3 && std::getline(csv_line_stream, csv_val, ','); ++j) {
            std::istringstream csv_val_stream(csv_val);
            if (!(csv_val_stream >> K[j][i - 4])) {
              break;
            }
            ++read_count;
          }
          if (read_count != 3) {
            throw std::runtime_error(
                  mayaCamReadingError("2", /* line= */ i + 1, mayacam, "There was an error reading 'camera matrix' values."));
          }
          break;
        }
        case 9: //R
        case 10:
        case 11:
        {
          int read_count = 0;
          for (int j = 0; j < 3 && std::getline(csv_line_stream, csv_val, ','); ++j) {
            std::istringstream csv_val_stream(csv_val);
            if (!(csv_val_stream >> rotation[j][i - 9])) {
              break;
            }
            ++read_count;
          }
          if (read_count != 3) {
            throw std::runtime_error(
                  mayaCamReadingError("2", /* line= */ i + 1, mayacam, "There was an error reading 'rotation' values."));
          }
          break;
        }
        case 14: //t
        case 15:
        case 16:
          if (!(csv_line_stream >> translation[i - 14])) {
            throw std::runtime_error(
                  mayaCamReadingError("2", /* line= */ i + 1, mayacam, "There was an error reading 'translation' values."));
            }
          ++translation_read_count;
          break;

      }
    }
    file.close();

    if (translation_read_count != 3) {
      throw std::runtime_error(
            mayaCamReadingError("2", /* line= */ 14 + 1, mayacam, "There was an error reading 'translation' values."));
    }

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
    calculateViewport(K[2][0], K[2][1], K[0][0], K[1][1]);

    // Calculate the image plane
    double z = -0.5* (K[0][0] + K[1][1]); // Average focal length, negated to be consistent with MayaCam 1.0
    calculateImagePlane(K[2][0], K[2][1], z);
  }

  void Camera::loadVTKCamera(const std::string& filename) {
    // Open and parse the file
    double version = -1.0, view_angle = -1.0, image_width = -1.0 , image_height = -1.0;
    double focal_point[3] = {-1.0, -1.0, -1.0}, camera_position[3] = { -1.0, -1.0, -1.0 }, view_up[3] = { -1.0, -1.0, -1.0 };
    std::fstream file(filename.c_str(), std::ios::in);
    if (!file.is_open()) {
      throw std::runtime_error("Error opening VTKCam file: " + filename);
    }
    std::string line;
    // The file is a series of key value pairs separated by a colon, # denotes a comment
    int line_num = 1;
    while (safeGetline(file, line)) {
      // Ignore comments and empty lines
      if (line.empty() || line[0] == '#') {
        line_num++;
        continue;
      }
      // Split the line into key and value
      std::string key, value;
      std::istringstream line_stream(line);
      if (!getline(line_stream, key, ':')) {
        file.close();
        throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing key."));
      }
      if (!getline(line_stream, value, ':')) {
        file.close();
        throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing value."));
      }
      // Parse the key value pair
      if (key == "version") {
        std::istringstream value_stream(value);
        if (!(value_stream >> version)) {
          file.close();
          throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing version number."));
        }
      }
      else if (key == "focal-point") {
        if (!parseArray(value, focal_point, 3)) {
          file.close();
          throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing focal-point."));
        }
      }
      else if (key == "camera-position") {
        if (!parseArray(value, camera_position, 3)) {
          file.close();
          throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing camera-position."));
        }
      }
      else if (key == "view-up") {
        if (!parseArray(value, view_up, 3)) {
          file.close();
          throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing view-up."));
        }
      }
      else if (key == "view-angle") {
        std::istringstream value_stream(value);
        if (!(value_stream >> view_angle)) {
          file.close();
          throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing view-angle."));
        }
      }
      else if (key == "image-width") {
        std::istringstream value_stream(value);
        if (!(value_stream >> image_width)) {
          file.close();
          throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing image-width."));
        }
      }
      else if (key == "image-height") {
        std::istringstream value_stream(value);
        if (!(value_stream >> image_height)) {
          file.close();
          throw std::runtime_error(vtkCamReadingError("1", line_num, filename, "Error parsing image-height."));
        }
      }
      line_num++;
    }

    // Close the file
    file.close();

    // Check that all the values were read
    if (version == -1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Missing version number."));
    }
    if (view_angle == -1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Missing view-angle."));
    }
    if (image_width == -1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Missing image-width."));
    }
    if (image_height == -1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Missing image-height."));
    }
    if (focal_point[0] == -1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Missing focal-point."));
    }
    if (camera_position[0] == -1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Missing camera-position."));
    }
    if (view_up[0] == -1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Missing view-up."));
    }

    // Check the version number
    if (version != 1.0) {
      throw std::runtime_error(vtkCamReadingError("1", -1, filename, "Unsupported version number."));
    }


    // Set the size
    size_[0] = image_width;
    size_[1] = image_height;

    Vec3d cam_pos(camera_position);
    Vec3d focal(focal_point);
    Vec3d up(view_up);
    double rot[9] = { 0.0 };
    calculateLookAtMatrix(cam_pos, focal, up, rot);
    coord_frame_ = CoordFrame(rot, cam_pos);

    // Calculate the focal length
    double focal_lengths[2] = { 0.0 };
    calculateFocalLength(view_angle, focal_lengths);

    // Calculate the principal point
    double cx = image_width / 2.0;
    double cy = image_height / 2.0;

    // Calculate the viewport
    calculateViewport(cx, cy , focal_lengths[0], focal_lengths[1]);

    // Calculate the image plane
    double z = -0.5* (focal_lengths[0] + focal_lengths[1]);
    calculateImagePlane(cx, cy, z);
  }

  void Camera::calculateViewport(const double& cx, const double& cy, const double& fx, const double& fy) {
    // Calculate the viewport

    // Validate that neither fx nor fy are zero
    if (fx == 0 || fy == 0) {
      throw std::runtime_error("Invalid camera parameters (fx or fy is zero)");
    }
    viewport_[0] = -(2.0f*cx - size_[0]) / fx;
    viewport_[1] = -(2.0f*cy - size_[1]) / fy;
    viewport_[2] = 2.0f* size_[0] / fx;
    viewport_[3] = 2.0f* size_[1] / fy;
  }

  void Camera::calculateImagePlane(const double& cx, const double& cy, const double& z) {
    // Pick a scale factor that places the image plane on the other side of the origin from the camera.
    double distance = sqrt(coord_frame_.translation()[0] * coord_frame_.translation()[0] +
      coord_frame_.translation()[1] * coord_frame_.translation()[1] +
      coord_frame_.translation()[2] * coord_frame_.translation()[2]);
    double scale = -1.5 * distance / z;

    double image_plane_trans[3];
    image_plane_trans[0] = scale * (size_[0] / 2.0 - cx);
    image_plane_trans[1] = scale * (size_[1] / 2.0 - cy);
    image_plane_trans[2] = scale * z;

    // Calculate the vertices at the corner of the image plane.
    double image_plane_center[3];
    coord_frame_.point_to_world_space(image_plane_trans, image_plane_center);

    double half_width = scale * size_[0] / 2.0;
    double half_height = scale * size_[1] / 2.0;

    double right[3] = { coord_frame_.rotation()[0],
      coord_frame_.rotation()[1],
      coord_frame_.rotation()[2] };
    double up[3] = { coord_frame_.rotation()[3],
      coord_frame_.rotation()[4],
      coord_frame_.rotation()[5] };

    image_plane_[0] = image_plane_center[0] - half_width * right[0] +
      half_height * up[0];
    image_plane_[1] = image_plane_center[1] - half_width * right[1] +
      half_height * up[1];
    image_plane_[2] = image_plane_center[2] - half_width * right[2] +
      half_height * up[2];

    image_plane_[3] = image_plane_center[0] - half_width * right[0] -
      half_height * up[0];
    image_plane_[4] = image_plane_center[1] - half_width * right[1] -
      half_height * up[1];
    image_plane_[5] = image_plane_center[2] - half_width * right[2] -
      half_height * up[2];

    image_plane_[6] = image_plane_center[0] + half_width * right[0] -
      half_height * up[0];
    image_plane_[7] = image_plane_center[1] + half_width * right[1] -
      half_height * up[1];
    image_plane_[8] = image_plane_center[2] + half_width * right[2] -
      half_height * up[2];

    image_plane_[9] = image_plane_center[0] + half_width * right[0] +
      half_height * up[0];
    image_plane_[10] = image_plane_center[1] + half_width * right[1] +
      half_height * up[1];
    image_plane_[11] = image_plane_center[2] + half_width * right[2] +
      half_height * up[2];
  }

  void Camera::calculateFocalLength(const double& view_angle, double focal_lengths[2]) {
    // Convert from deg to rad
    double angle_rad = view_angle * (M_PI / 180);

    focal_lengths[0] = size_[0] / (2 * std::tan(angle_rad / 2));
    focal_lengths[1] = size_[1] / (2 * std::tan(angle_rad / 2));
  }

  void Camera::calculateLookAtMatrix(const Vec3d& eye, const Vec3d& center, const Vec3d& up, double matrix[9]) {
    // Implementation based off of:
    // https://www.khronos.org/opengl/wiki/GluLookAt_code
    Vec3d forward = unit(center - eye);
    // This vector points to the right-hand side of the camera's orientation.
    Vec3d side = unit(cross(forward, up));
    Vec3d perpendicularUp = cross(side, forward);
    matrix[0] = side.x;
    matrix[1] = side.y;
    matrix[2] = side.z;
    matrix[3] = perpendicularUp.x;
    matrix[4] = perpendicularUp.y;
    matrix[5] = perpendicularUp.z;
    matrix[6] = -forward.x;
    matrix[7] = -forward.y;
    matrix[8] = -forward.z;
  }

} // namespace XROMM
