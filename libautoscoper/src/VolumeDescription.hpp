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

/// \file VolumeDescription.hpp
/// \author Andy Loomis, Mark Howison, Benjamin Knorlein

#ifndef XROMM_GPU_VOLUME_DESCRIPTION_HPP
#define XROMM_GPU_VOLUME_DESCRIPTION_HPP

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
struct cudaArray;
typedef cudaArray Image;
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
#  include "gpu/opencl/OpenCL.hpp"
#endif
#include "Vector.hpp"

namespace xromm {
class Volume;

namespace gpu {

// The VolumeDescription class provides an abstraction between the volume as it
// is stored in CPU memory and the volume as it is stored in GPU memory. When it
// is created from a Volume, it automatically crops the volume to a minimal axis
// aligned bounding box so that as little GPU memory is wasted as possible.

class VolumeDescription
{
public:
  VolumeDescription(const Volume& volume);
  ~VolumeDescription();

  const float* invScale() const { return invScale_; }
  const float* invTrans() const { return invTrans_; }
  const double* transCenter() const { return transCenter_; }
  Vec3f transCenterVec() { return Vec3f(transCenter_); }

  float minValue() const { return minValue_; }
  float maxValue() const { return maxValue_; }

  const Image* image() const { return image_; }

private:
  VolumeDescription(const VolumeDescription&);
  VolumeDescription& operator=(const VolumeDescription&);

  float minValue_;
  float maxValue_;
  float invScale_[3];
  float invTrans_[3];
  double transCenter_[3];

  Image* image_;
};
} // namespace gpu
} // namespace xromm

#endif // XROMM_GPU_VOLUME_DESCRIPTION_HPP
