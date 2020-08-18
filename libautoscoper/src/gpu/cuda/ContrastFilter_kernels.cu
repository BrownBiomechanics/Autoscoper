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

/// \file ContrastFilter_kernels.cu
/// \author Andy Loomis

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ContrastFilter_kernels.h"
#include <cutil_math.h>

__global__ void contrast_filter_kernel(const float* input, float* output,
                            int width, int height,
                            float alpha, float beta, int size);

namespace xromm { namespace gpu {

void contrast_filter_apply(const float* input, float* output,
                           int width, int height,
                           float alpha, float beta, int size)
{
    dim3 blockDim(32, 32);
    dim3 gridDim((width+blockDim.x-1)/blockDim.x,
                 (height+blockDim.y-1)/blockDim.y);
    
    contrast_filter_kernel <<<gridDim, blockDim>>> (input, output,
                                                  width, height,
                                                  alpha, beta, size);
}

} } // namespace xromm::cuda

__device__ float average(const float* input, int width, int height, int x, int y, int size)
{
    float n = 0.0f;
    float sum = 0.0f;
    int minI = max(y-size/2, 0);
    int maxI = min(y+(size+1)/2, height);
    int minJ = max(x-size/2, 0);
    int maxJ = min(x+(size+1)/2, width);
    for (int i = minI; i < maxI; ++i) {
        for (int j = minJ; j < maxJ; ++j) {
            n += 1.0f;
            sum += input[i*width+j];
        }
    }
    return sum/n;
}

__global__ void contrast_filter_kernel(const float* input, float* output,
                            int width, int height,
                            float alpha, float beta, int size)
{
    short x = blockIdx.x*blockDim.x+threadIdx.x;
    short y = blockIdx.y*blockDim.y+threadIdx.y;

    if (x > width-1 || y > height-1) {
        return;
    }

    float fxy = input[y*width+x];
    float axy = average(input, width, height, x, y, size);
    float gxy = 0.0f;
    if (axy > 0.01f) {
        gxy = pow(axy,alpha-beta)*pow(fxy,beta);
    }
    output[y*width+x] = gxy;
}
