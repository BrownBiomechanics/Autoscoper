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

/// \file SharpenFilter_kernels.cu
/// \author Emily Fu

#include "SharpenFilter_kernels.h"
#include "stdlib.h"
#include "math.h"

__global__
void sharpen_filter_kernel(const float* input, float* output,
                           int width, int height,
                           float* filter, int filterSize, float contrast, float threshold);

namespace xromm { namespace gpu {
void sharpen_filter_apply(const float* input, float* output,
                          int width, int height, float* sharpen, int filterSize, float contrast, float threshold)
{
  dim3 blockDim(32, 32);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);


  sharpen_filter_kernel << < gridDim, blockDim >> > (input, output,
                                                     width, height,
                                                     sharpen, filterSize, contrast, threshold);

}
} } // namespace xromm::cuda

static __device__
float filterConvolution(const float* input, int width, int height, int x, int y, float* filter, int filterSize) // returns blurred pixel for comparison with original
{

  float centerValue = 0.0f;
  int filterRadius = (filterSize - 1) / 2;

  for (int i = 0; i < filterSize; ++i) {
    for (int j = 0; j < filterSize; ++j) {

      int a = x - filterRadius + i;
      int b = y - filterRadius + j;

      if (!(a < 0 || a >= width || b < 0 || b >= height))
        centerValue = centerValue + (filter[i * filterSize + j]) * (input[b * width + a]);
    }
  }


  return centerValue;
}

__global__
void sharpen_filter_kernel(const float* input, float* output,
                           int width, int height,
                           float* filter, int filterSize, float contrast, float threshold)
{
  short x = blockIdx.x * blockDim.x + threadIdx.x;
  short y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > width - 1 || y > height - 1) {
    return;
  }

  float blur = filterConvolution(input, width, height, x, y, filter, filterSize);

// if original pixel and blurred pixel differ by more than threshold, difference is adjusted by contrast and added to original, else no change
  if (abs(input[y * width + x] - blur) > threshold) {
    output[y * width + x] = input[y * width + x] + contrast * (input[y * width + x] - blur);

    if (output[y * width + x]  > 1)
      output[y * width + x] = 1;
    if (output[y * width + x]  < 0)
      output[y * width + x] = 0;
  } else
    output[y * width + x] = input[y * width + x];

}
