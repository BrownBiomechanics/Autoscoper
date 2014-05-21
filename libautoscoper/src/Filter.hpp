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

/// \file Filter.hpp
/// \author Andy Loomis, Mark Howison

#ifndef XROMM_GPU_FILTER_HPP
#define XROMM_GPU_FILTER_HPP

#include <string>

#ifdef WITH_CUDA
typedef float Buffer;
#else
#include "gpu/opencl/OpenCL.hpp"
#endif



namespace xromm { namespace gpu {

class Filter
{
public:
    enum
    {
        XROMM_GPU_CONTRAST_FILTER,
        XROMM_GPU_SOBEL_FILTER,
        XROMM_GPU_MEDIAN_FILTER,
        XROMM_GPU_GAUSSIAN_FILTER,
        XROMM_GPU_SHARPEN_FILTER
    };

    Filter(int type, const std::string& name)
        : type_(type), name_(name), enabled_(true) {}

	virtual ~Filter() {}

    // Apply the filter to the input image
    virtual void apply(const Buffer* input,
                       Buffer* output,
                       int width,
                       int height) = 0;


    // Accessors and mutators
    int type() const { return type_; }

    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    bool enabled() const { return enabled_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

protected:
    int type_;
    std::string name_;
    bool enabled_;
};

} } // namespace xromm::opencl

#endif // XROMM_GPU_FILTER_HPP
