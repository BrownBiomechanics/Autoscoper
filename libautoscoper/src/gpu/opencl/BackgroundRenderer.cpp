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

/// \file BackgroundRenderer.cpp
/// \author Ben Knorlein

#include <iostream>
#include <sstream>

#include "BackgroundRenderer.hpp"

#define BX 16
#define BY 16

using namespace std;

namespace xromm { namespace gpu
{

#include "gpu/opencl/kernel/BackgroundRenderer.cl.h"

static Program background_renderer_program_;


BackgroundRenderer::BackgroundRenderer() : image_(0)
{
    image_plane_[0] = -1.0f;
    image_plane_[1] = -1.0f;
    image_plane_[2] =  2.0f;
    image_plane_[3] =  2.0f;

    viewport_[0] = -1.0f;
    viewport_[1] = -1.0f;
    viewport_[2] =  2.0f;
    viewport_[3] =  2.0f;
}

BackgroundRenderer::~BackgroundRenderer()
{
  if (image_) delete image_;
}

void
BackgroundRenderer::set_back(const void* data, size_t width, size_t height)
{
  cl_image_format format;
  format.image_channel_order = CL_R;
  format.image_channel_data_type = CL_FLOAT;


  if (image_) delete image_;
  size_t dims[3] = { width, height, 1 };
  image_ = new Image(dims, &format, CL_MEM_READ_ONLY);
  image_->read(data);
}

void
BackgroundRenderer::set_image_plane(float x, float y, float width, float height)
{
    image_plane_[0] = x;
    image_plane_[1] = y;
    image_plane_[2] = width;
    image_plane_[3] = height;
}

void
BackgroundRenderer::set_viewport(float x, float y, float width, float height)
{
    viewport_[0] = x;
    viewport_[1] = y;
    viewport_[2] = width;
    viewport_[3] = height;
}

void
BackgroundRenderer::render(const Buffer* buffer, unsigned width, unsigned height, float threshold) const
{
  Kernel* kernel = background_renderer_program_.compile(
        BackgroundRenderer_cl, "background_render_kernel");

  kernel->addBufferArg(buffer);
  kernel->addArg(width);
  kernel->addArg(height);
  kernel->addArg(image_plane_[0]);
  kernel->addArg(image_plane_[1]);
  kernel->addArg(image_plane_[2]);
  kernel->addArg(image_plane_[3]);
  kernel->addArg(viewport_[0]);
  kernel->addArg(viewport_[1]);
  kernel->addArg(viewport_[2]);
  kernel->addArg(viewport_[3]);
  kernel->addArg(threshold),
  kernel->addImageArg(image_);

    // Calculate the block and grid sizes.
  kernel->block2d(BX, BY);
    kernel->grid2d((width+BX-1)/BX, (height+BY-1)/BY);

  kernel->launch();

  delete kernel;
}

} } // namespace xromm::opencl
