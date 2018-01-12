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

/// \file View.hpp
/// \author Andy Loomis, Mark Howison

#ifndef XROMM_GPU_VIEW_HPP
#define XROMM_GPU_VIEW_HPP

#include <vector>
#include <string>
#ifdef WITH_CUDA
typedef float Buffer;
typedef float GLBuffer;
#else
#include "gpu/opencl/OpenCL.hpp"
#endif

namespace xromm
{

class Camera;

namespace gpu
{

class RayCaster;
class RadRenderer;
class Filter;

// This class encapsulates everything related to the rendering of the drrs and
// radiographs. It contains the cameras, renderers, adn filters. The renderers
// and filters have quite a few parameters that effect how the rendering looks,
// and those can be modified at any time by the front end application.

class View
{
public:
    // Constructs a View from a camera
    View(Camera& camera);
    ~View();

private:
    View(const View& view);
    View& operator=(const View& view);

public:
    // Accessors
    Camera* camera() { return camera_; }
    const Camera* camera() const { return camera_; }
	RayCaster* drrRenderer(int idx) { return drrRenderer_[idx]; }
	const RayCaster* drrRenderer(int idx) const { return drrRenderer_[idx]; }
    RadRenderer* radRenderer() { return radRenderer_; }
    const RadRenderer* radRenderer() const { return radRenderer_; }
    std::vector<Filter*>& drrFilters() { return drrFilters_; }
    const std::vector<Filter*>& drrFilters() const { return drrFilters_; }
    std::vector<Filter*>& radFilters() { return radFilters_; }
    const std::vector<Filter*>& radFilters() const { return radFilters_; }

	void addDrrRenderer();
	void saveImage(std::string filename, int width, int height);

    // Rendering functions
    void renderRad(Buffer* buffer, unsigned int width, unsigned int height);
    void renderRad(unsigned int pbo, unsigned int width, unsigned int height);

    void renderDrr(Buffer* buffer, unsigned int width, unsigned int height);
    void renderDrr(unsigned int  pbo, unsigned int width, unsigned int height);

    void render(GLBuffer* buffer, unsigned int width, unsigned int height);
    void render(unsigned int  pbo, unsigned int width, unsigned int height);

    bool drr_enabled;
    bool rad_enabled;

private:
#ifdef WITH_CUDA
	void init();
#else
	void init(unsigned width, unsigned height);
#endif

    void filter(const std::vector<Filter*>& filters,
                const Buffer* input,
                Buffer* output,
                unsigned width,
                unsigned height);

    Camera* camera_;
	std::vector <RayCaster*> drrRenderer_;
    RadRenderer* radRenderer_;

    std::vector<Filter*> drrFilters_;
    std::vector<Filter*> radFilters_;

    size_t maxWidth_;
	size_t maxHeight_;

	std::vector<Buffer*> drrBuffer_;
	Buffer* drrBufferMerged_;
    Buffer* drrFilterBuffer_;
    Buffer* radBuffer_;
    Buffer* radFilterBuffer_;
    Buffer* filterBuffer_;

	bool inited_;
};

} } //namespace xromm::opencl

#endif // XROMM_GPU_VIEW_HPP
