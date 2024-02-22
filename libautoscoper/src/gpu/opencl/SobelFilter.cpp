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

/// \file SobelFilter.cpp
/// \author Andy Loomis, Mark Howison
//
#include <sstream>
#include "SobelFilter.hpp"

#define BX 16
#define BY 16

namespace xromm {
namespace gpu {
#include "gpu/opencl/kernel/SobelFilter.cl.h"

static Program sobel_program_;

static int num_sobel_filters = 0;

SobelFilter::SobelFilter()
  : Filter(XROMM_GPU_SOBEL_FILTER, "")
  , scale_(1.0f)
  , blend_(0.5f)
{
  std::stringstream name_stream;
  name_stream << "SobelFilter" << (++num_sobel_filters);
  name_ = name_stream.str();
}

void SobelFilter::apply(const Buffer* input, Buffer* output, int width, int height)
{
  Kernel* kernel = sobel_program_.compile(SobelFilter_cl, "sobel_filter_kernel");

  kernel->block2d(BX, BY);
  kernel->grid2d((width - 1) / BX + 1, (height - 1) / BY + 1);

  kernel->addBufferArg(input);
  kernel->addBufferArg(output);
  kernel->addArg(width);
  kernel->addArg(height);
  kernel->addArg(scale_);
  kernel->addArg(blend_);

  kernel->launch();

  delete kernel;
}
} // namespace gpu
} // namespace xromm
