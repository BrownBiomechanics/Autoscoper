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

/// \file RadRenderer.cpp
/// \author Andy Loomis

#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#include "RadRenderer.hpp"
#include "RadRenderer_kernels.h"

namespace xromm { namespace gpu
{

static int num_rad_renderers = 0;

RadRenderer::RadRenderer() : array_(0)
{
    image_plane_[0] = -1.0f;
    image_plane_[1] = -1.0f;
    image_plane_[2] =  2.0f;
    image_plane_[3] =  2.0f;

    viewport_[0] = -1.0f;
    viewport_[1] = -1.0f;
    viewport_[2] =  2.0f;
    viewport_[3] =  2.0f;

    std::stringstream name_stream;
    name_stream << "RadRenderer" << (++num_rad_renderers);
    name_ = name_stream.str();
}

RadRenderer::~RadRenderer()
{
    num_rad_renderers = 0;
    cutilSafeCall(cudaFreeArray(array_));
}

void
RadRenderer::set_rad(const void* data, size_t width, size_t height, size_t bps)
{
    cutilSafeCall(cudaFreeArray(array_));

    // Create a 2D array.
    cudaChannelFormatDesc desc;
    switch (bps) {
        case 8:  desc = cudaCreateChannelDesc<unsigned char>(); break;
        case 16: desc = cudaCreateChannelDesc<unsigned short>(); break;
        case 32: desc = cudaCreateChannelDesc<unsigned int>(); break;
        default:
            std::cerr << "RadRenderer::rad(): Unsupported bit depth "
                 << bps << std::endl;
            return;
    }
    cutilSafeCall(cudaMallocArray(&array_, &desc, width, height));

    // Copy data to 2D array.
    cutilSafeCall(cudaMemcpyToArray(array_,
                                    0,
                                    0,
                                    data,
                                    width*height*(bps/8),
                                    cudaMemcpyHostToDevice));
}

void
RadRenderer::set_image_plane(float x, float y, float width, float height)
{
    image_plane_[0] = x;
    image_plane_[1] = y;
    image_plane_[2] = width;
    image_plane_[3] = height;
}

void
RadRenderer::set_viewport(float x, float y, float width, float height)
{
    viewport_[0] = x;
    viewport_[1] = y;
    viewport_[2] = width;
    viewport_[3] = height;
}

void
RadRenderer::render(float* buffer, size_t width, size_t height) const
{
    video_bind_array(array_);
    video_render(buffer,
                 (int)width,
                 (int)height,
                 image_plane_[0],
                 image_plane_[1],
                 image_plane_[2],
                 image_plane_[3],
                 viewport_[0],
                 viewport_[1],
                 viewport_[2],
                 viewport_[3]);
}

} } // namespace xromm::cuda
