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

using namespace std;

namespace xromm
{

Camera::Camera(const string& mayacam) : mayacam_(mayacam)
{
    // Load the mayacam.csv file into an array of doubles

    fstream file(mayacam.c_str(), ios::in);
    if (file.is_open() == false) {
        throw runtime_error("File not found: " + mayacam);
    }

    double csv_vals[5][3];
    string csv_line, csv_val;
    for (int i = 0; i < 5 && getline(file, csv_line); ++i) {
        istringstream csv_line_stream(csv_line);
        for (int j = 0; j < 3 && getline(csv_line_stream,csv_val,','); ++j) {
            istringstream csv_val_stream(csv_val);
            if (!(csv_val_stream >> csv_vals[i][j])) {
                throw runtime_error("Invalid MayaCam file");
            }
        }
    }

    // Line 1: Camera location in world space
    // Line 2: Rotations around the local x, y, and z axes of the camera. The
    //         order of rotation is z, y, then x.
    // Line 3: Position of the film plane relative to the camera. Given in the
    //         camera's local space. These values are calculated using the
    //         values in lines 4 and 5.
    // Line 4: u0, v0, Z
    // Line 5: scale, image_width, image_height
    //
    // If the image width and height are set to 0 they are assumed to be 1024.
    //
    // The equations used to calculate the position and size of the film plane
    // are as follows:
    //
    // image_plane_x = scale*(image_width/2-u0)
    // image_plane_y = scale*(image_height/2-v0)
    // image_plane_z = -scale*Z
    //
    // image_plane_width = scale*image_width
    // image_plane_height = scale*image_height
    //
    // All units are in centimeters

    double* translation = csv_vals[0];
    double* rotation = csv_vals[1];
    double* image_plane_trans = csv_vals[2];

    double u0 = csv_vals[3][0];
    double v0 = csv_vals[3][1];
    double z = csv_vals[3][2];

    // Default to 1024x1024
    double image_width = csv_vals[4][1];
    if (image_width == 0) image_width = 1024;

    double image_height = csv_vals[4][2];
    if (image_height == 0) image_height = 1024;

    // Calculate the cameras local coordinate frame
    double xyzypr[6] = {translation[0],translation[1],translation[2],
                        rotation[2],rotation[1],rotation[0]};
    coord_frame_ = CoordFrame::from_xyzypr(xyzypr);

    // Calculate the viewport
    viewport_[0] = (2.0f*u0-image_width)/z;
    viewport_[1] = (2.0f*v0-image_height)/z;
    viewport_[2] = -2.0f*image_width/z;
    viewport_[3] = -2.0f*image_height/z;

    // Choose the scaling factor such that the image plane will be on the
    // other side of the origin from the camera. The values in the mayacam
    // file are discarded.
    double distance = sqrt(translation[0]*translation[0]+
                           translation[1]*translation[1]+
                           translation[2]*translation[2]);
    double scale = -1.5*distance/z;

    image_plane_trans[0] = scale*(image_width/2.0-u0);
    image_plane_trans[1] = scale*(image_height/2.0-v0);
    image_plane_trans[2] = scale*z;

    // Calculate the vertices at the corner of the image plane.
    double image_plane_center[3];
    coord_frame_.point_to_world_space(image_plane_trans, image_plane_center);

    double half_width  = scale*image_width/2.0;
    double half_height = scale*image_height/2.0;

    double right[3] = {coord_frame_.rotation()[0],
                       coord_frame_.rotation()[1],
                       coord_frame_.rotation()[2]};
    double up[3] = {coord_frame_.rotation()[3],
                    coord_frame_.rotation()[4],
                    coord_frame_.rotation()[5]};

    image_plane_[0]  = image_plane_center[0]-half_width*right[0]+
                       half_height*up[0];
    image_plane_[1]  = image_plane_center[1]-half_width*right[1]+
                       half_height*up[1];
    image_plane_[2]  = image_plane_center[2]-half_width*right[2]+
                       half_height*up[2];

    image_plane_[3]  = image_plane_center[0]-half_width*right[0]-
                       half_height*up[0];
    image_plane_[4]  = image_plane_center[1]-half_width*right[1]-
                       half_height*up[1];
    image_plane_[5]  = image_plane_center[2]-half_width*right[2]-
                       half_height*up[2];

    image_plane_[6]  = image_plane_center[0]+half_width*right[0]-
                       half_height*up[0];
    image_plane_[7]  = image_plane_center[1]+half_width*right[1]-
                       half_height*up[1];
    image_plane_[8]  = image_plane_center[2]+half_width*right[2]-
                       half_height*up[2];

    image_plane_[9]  = image_plane_center[0]+half_width*right[0]+
                       half_height*up[0];
    image_plane_[10] = image_plane_center[1]+half_width*right[1]+
                       half_height*up[1];
    image_plane_[11] = image_plane_center[2]+half_width*right[2]+
                       half_height*up[2];

    // Choose appropriate values for the fov, ratio, near and far clip plans
    double max_width  = half_width+fabs(image_plane_trans[0]);
    double max_height = half_height+fabs(image_plane_trans[1]);

    field_of_view_   = 360.0/M_PI*atan(max_height/fabs(image_plane_trans[2]));
    aspect_ratio_ = max_width/max_height;
}

} // namespace XROMM
