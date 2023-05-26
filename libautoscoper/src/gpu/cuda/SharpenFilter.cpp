// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provideId that the following conditions are
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

/// \file SharpenFilter.cpp
/// \author Emily Fu

#include "SharpenFilter.hpp"
#include "SharpenFilter_kernels.h"

#include <sstream>
#include <iostream>
#include "stdlib.h"
#include "math.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace xromm { namespace gpu {

// Unique identifier for each contrast filter

static int num_sharpen_filters = 0;

SharpenFilter::SharpenFilter()
    : Filter(XROMM_GPU_SHARPEN_FILTER,""),
      radius_(1),
      contrast_(1),
      sharpen_(NULL)
{
    std::stringstream name_stream;
    name_stream << "SharpenFilter" << (++num_sharpen_filters);
    name_ = name_stream.str();

//default values--threshold = 0 so all pixels are sharpened
    set_radius(1);
    set_contrast(1);
    set_threshold(0);
}

SharpenFilter::~SharpenFilter()
{
     cudaFree(sharpen_);
}

void SharpenFilter::set_radius(float radius)
{
    if (radius < 0)
        radius = 0;

    radius_ = radius;

    makeFilter();

}

void SharpenFilter::set_contrast(float contrast)
{
    if(contrast<1)
        contrast = 1;

    contrast_ = contrast;
}

void SharpenFilter::set_threshold(float threshold)
{
    threshold_ = threshold;

}

void SharpenFilter::makeFilter() //makes a Gaussian blur filter (filterSize*filterSize) with stdev radius_
{

    int filterRadius= 3*radius_;
    filterSize_ = 2*filterRadius + 1;

    if(filterSize_ == 1)
        return;

    float* sharpen = (float *)malloc(sizeof(float )*(filterSize_*filterSize_));

    float sum = 0.0f;

    for(int i = 0; i < filterSize_; ++i){
        for(int j = 0; j < filterSize_ ; ++j){
            sharpen[i*filterSize_+j] = pow((double) 2.71828,(-( pow((double) (i-filterRadius),2) +pow((double) (j-filterRadius), 2) ) / (2* radius_))); //equation for Gaussian at (i, j)
            sum = sum +  sharpen[i*filterSize_ +j];
        }
    }

    float temp = 0.0f;

//normalize the filter

    for(int i = 0 ; i < filterSize_; ++i){
        for(int j = 0 ; j < filterSize_; ++j) {
            temp = sharpen[i*filterSize_ +j];
            sharpen[i*filterSize_ + j] = temp / sum;
         }
    }

//copy the filter to GPU

    float * sharpenGPU;
    cudaMalloc(&sharpenGPU, sizeof(float )*(filterSize_*filterSize_));
    cudaMemcpy(sharpenGPU, sharpen, (sizeof(float )*(filterSize_*filterSize_)),cudaMemcpyHostToDevice);

    free(sharpen);
    cudaFree(sharpen_);

    sharpen_ = sharpenGPU;
}

void

SharpenFilter::apply(const float* input,
                      float* output,
                      int width,
                      int height)
{
    if(filterSize_ == 1 ) //if filterSize_ = 1, filter does not change image
       cudaMemcpy(output, input, (sizeof(float )*(filterSize_*filterSize_)), cudaMemcpyDeviceToDevice);
    else
      sharpen_filter_apply(input,output,width,height, sharpen_, filterSize_, contrast_, threshold_);
}

} } // namespace xromm::cuda
