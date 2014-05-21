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

/// \file Sobel_kernels.cu
/// \author Andy Loomis

#include "SobelFilter_kernels.h"

__global__
void sobel_filter_kernel(const float* input, float* output, int width, int height,
                         float scale, float blend);

namespace xromm
{

namespace gpu
{

void sobel_filter(const float* input, float* output, int width, int height,
                  float scale)
{
    sobel_filter_blend(input, output, width, height, scale, 0.0f);
}

void sobel_filter_blend(const float* input, float* output, int width,
                        int height, float scale, float blend)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((width+blockDim.x-1)/blockDim.x,
                 (height+blockDim.y-1)/blockDim.y);
    
    sobel_filter_kernel<<<gridDim, blockDim>>>(input, output, width,
                                               height, scale, blend);
}

} // namespace gpu

} // namespace xromm

__global__
void sobel_filter_kernel(const float* input,
                         float* output,
                         int width,
                         int height,
                         float scale,
                         float blend)
{
    short x1 = blockIdx.x*blockDim.x+threadIdx.x;
    short y1 = blockIdx.y*blockDim.y+threadIdx.y;
   
    if (x1 > width-1 || y1 > height-1) {
        return;
    }

    short x0 = x1-1; if (x0 < 0) x0 = 0;
    short y0 = y1-1; if (y0 < 0) y0 = 0;
    
    short x2 = x1+1; if (x2 > width-1) x2 = width-1;
    short y2 = y1+1; if (y2 > height-1) y2 = height-1;

    float pix00 = input[y0*width+x0];
    float pix01 = input[y0*width+x1];
    float pix02 = input[y0*width+x2];
    float pix10 = input[y1*width+x0];
    float pix11 = input[y1*width+x1];
    float pix12 = input[y1*width+x2];
    float pix20 = input[y2*width+x0];
    float pix21 = input[y2*width+x1];
    float pix22 = input[y2*width+x2];
    
    float horz = pix02+2*pix12+pix22-pix00-2*pix10-pix20;
    float vert = pix00+2*pix01+pix02-pix20-2*pix21-pix22;
    float grad = sqrt(horz*horz+vert*vert);

    float sum;
    if (blend < 0.5f) {
        sum = scale*grad+2.0f*blend*pix11;
    }
    else {
        sum = 2.0f*(1.0f-blend)*scale*grad+pix11;
    }
    if (sum < 0.0f) {
        sum = 0.0f;
    }
    else if (sum > 1.0f) {
        sum = 1.0f;
    }
    
    output[y1*width+x1] = sum;
}
