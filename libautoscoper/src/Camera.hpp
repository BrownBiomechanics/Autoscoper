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

/// \file Camera.hpp
/// \author Andy Loomis

#ifndef XROMM_CAMERA_HPP
#define XROMM_CAMERA_HPP

#include <string>

#include "CoordFrame.hpp"

namespace xromm
{

// Simple class encapsulating an xromm camera. This class is constructed from a
// mayacam file, which describes the orientation and position of the camera, and
// is created during the registration phase of the xromm matlab pipeline.

class Camera
{
public:

    // Loads a mayaCam.csv file

    Camera(const std::string& mayacam);

    // Accessors

    const std::string& mayacam() const { return mayacam_; }

    const CoordFrame& coord_frame() const { return coord_frame_; }

    const double* image_plane() const { return image_plane_; }

    const double* viewport() const { return viewport_; }

  const double* size() const { return size_; }

private:

    std::string mayacam_; // filename

    CoordFrame coord_frame_;

    double image_plane_[12];

    double viewport_[4];

  double size_[2];

  void loadMayaCam1(const std::string& mayacam);

  void loadMayaCam2(const std::string& mayacam);

};

} // namespace xromm

#endif // XROMM_CAMERA_HPP
