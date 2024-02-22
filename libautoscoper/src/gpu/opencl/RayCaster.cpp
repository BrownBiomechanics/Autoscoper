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

/// \file RayCaster.cpp
/// \author Andy Loomis, Mark Howison, Benjamin Knorlein

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "RayCaster.hpp"
#include "VolumeDescription.hpp"

#define BX 16
#define BY 16

namespace xromm { namespace gpu {
#include "gpu/opencl/kernel/RayCaster.cl.h"

static Program raycaster_program_;

static int num_ray_casters = 0;

RayCaster::RayCaster() : volumeDescription_(0),
                         sampleDistance_(0.5f),
                         rayIntensity_(10.f),
                         cutoff_(0.0f),
                         name_("")
{
  std::stringstream name_stream;
  name_stream << "DrrRenderer" << (++num_ray_casters);
  name_ = name_stream.str();

  viewport_[0] = -1.0f;
  viewport_[1] = -1.0f;
  viewport_[2] =  2.0f;
  viewport_[3] =  2.0f;

  b_viewport_ = new Buffer(4 * sizeof(float), CL_MEM_READ_ONLY);
  b_viewport_->read(viewport_);
  visible_ = true;
}

RayCaster::~RayCaster()
{
  if (b_viewport_) delete b_viewport_;
}

void
RayCaster::setVolume(VolumeDescription& volume)
{
  volumeDescription_ = &volume;
}

void
RayCaster::setInvModelView(const double* invModelView)
{
  if (!volumeDescription_) {
    std::cerr << "RayCaster: ERROR: Unable to calculate matrix." << std::endl;
    exit(0);
  }

  const float* invScale = volumeDescription_->invScale();
  const float* invTrans = volumeDescription_->invTrans();

  // clang-format off
  invModelView_[0]  = invModelView[0] * invScale[0] +
                      invModelView[12] * invTrans[0];
  invModelView_[1]  = invModelView[1] * invScale[0] +
                      invModelView[13] * invTrans[0];
  invModelView_[2]  = invModelView[2] * invScale[0] +
                      invModelView[14] * invTrans[0];
  invModelView_[3]  = invModelView[3] * invScale[0] +
                      invModelView[15] * invTrans[0];
  invModelView_[4]  = invModelView[4] * invScale[1] +
                      invModelView[12] * invTrans[1];
  invModelView_[5]  = invModelView[5] * invScale[1] +
                      invModelView[13] * invTrans[1];
  invModelView_[6]  = invModelView[6] * invScale[1] +
                      invModelView[14] * invTrans[1];
  invModelView_[7]  = invModelView[7] * invScale[1] +
                      invModelView[15] * invTrans[1];
  invModelView_[8]  = invModelView[8] * invScale[2] +
                      invModelView[12] * invTrans[2];
  invModelView_[9]  = invModelView[9] * invScale[2] +
                      invModelView[13] * invTrans[2];
  invModelView_[10] = invModelView[10] * invScale[2] +
                      invModelView[14] * invTrans[2];
  invModelView_[11] = invModelView[11] * invScale[2] +
                      invModelView[15] * invTrans[2];
  invModelView_[12] = invModelView[12];
  invModelView_[13] = invModelView[13];
  invModelView_[14] = invModelView[14];
  invModelView_[15] = invModelView[15];
  // clang-format on

#if DEBUG
  // clang-format off
  fprintf(stdout, "RayCaster: new invModelView:\n %f, %f, %f, %f\n %f, %f, %f, %f\n %f, %f, %f, %f\n %f, %f, %f, %f\n",
          invModelView[0], invModelView[1], invModelView[2], invModelView[3],
          invModelView[4], invModelView[5], invModelView[6], invModelView[7],
          invModelView[8], invModelView[9], invModelView[10], invModelView[11],
          invModelView[12], invModelView[13], invModelView[14], invModelView[15]);
  // clang-format on
#endif
}

void
RayCaster::setViewport(float x, float y, float width, float height)
{
  viewport_[0] = x;
  viewport_[1] = y;
  viewport_[2] = width;
  viewport_[3] = height;

#if DEBUG
  fprintf(stdout, "RayCaster: new viewport: %f, %f, %f, %f\n",
          viewport_[0], viewport_[1], viewport_[2], viewport_[3]);
#endif

  b_viewport_->read(viewport_);
}

void
RayCaster::render(const Buffer* buffer, unsigned width, unsigned height)
{
  if (!volumeDescription_) {
    std::cerr << "RayCaster: WARNING: No volume loaded." << std::endl;
    return;
  }

  if (!visible_) {
    buffer->fill((char)0x00);
    return;
  }

  Kernel* kernel = raycaster_program_.compile(
    RayCaster_cl, "volume_render_kernel");

  Buffer* b_imv = new Buffer(12 * sizeof(float), CL_MEM_READ_ONLY);
  b_imv->read(invModelView_);

  // Calculate the block and grid sizes.
  kernel->block2d(BX, BY);
  kernel->grid2d((width + BX - 1) / BX, (height + BY - 1) / BY);

  kernel->addBufferArg(buffer);
  kernel->addArg(width);
  kernel->addArg(height);
  kernel->addArg(sampleDistance_);
  kernel->addArg(rayIntensity_);
  kernel->addArg(cutoff_);
  kernel->addBufferArg(b_viewport_);
  kernel->addBufferArg(b_imv);
  kernel->addImageArg(volumeDescription_->image());

  kernel->launch();

  delete kernel;
  delete b_imv;
}
} } // namespace xromm::opencl

