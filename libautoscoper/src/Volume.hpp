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

/// \file Volume.hpp
/// \author Andy Loomis

#ifndef XROMM_VOLUME_HPP
#define XROMM_VOLUME_HPP

#include <string>

namespace xromm
{

// This class represents a volumetric image. It is created from a single tiff
// file, which stores a stack of tiff images. It has some immutable properties,
// like width, height, depth, and bits per sample bps, but it also contains
// important metadata about the volume, such as the scale of the voxels in
// millimeters, and the order in which to interpret the stacks.

class Volume
{
public:

  // Loads a tiff volume

  Volume(const std::string& filename);

  Volume(const Volume& volume);

  ~Volume();

  Volume& operator=(const Volume&);

public:

  // Accessors

  const std::string& name() const { return name_; }

  size_t width() const { return width_; }

  size_t height() const { return height_; }

  size_t depth() const { return depth_; }

  size_t bps() const { return bps_; }

  const void* data() const { return data_; }

  // Volume properties that also affect rendering

  float scaleX() const { return scaleX_; }

  void scaleX(float scale) { scaleX_ = scale; }

  float scaleY() const { return scaleY_; }

  void scaleY(float scale) { scaleY_ = scale; }

  float scaleZ() const { return scaleZ_; }

  void scaleZ(float scale) { scaleZ_ = scale; }

  bool flipX() const { return flipX_; }

  void flipX(bool flip) { flipX_ = flip; }

  bool flipY() const { return flipY_; }

  void flipY(bool flip) { flipY_ = flip; }

  bool flipZ() const { return flipZ_; }

  void flipZ(bool flip) { flipZ_ = flip; }

private:

  std::string name_;

  size_t width_;

  size_t height_;

  size_t depth_;

  size_t bps_;

  void* data_;

  float scaleX_;

  float scaleY_;

  float scaleZ_;

  bool flipX_;

  bool flipY_;

  bool flipZ_;
};
} // namespace xromm

#endif // XROMM_VOLUME_HPP
