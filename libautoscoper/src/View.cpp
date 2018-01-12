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

/// \file View.cpp
/// \author Andy Loomis, Mark Howison

#include "View.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>

#include "Camera.hpp"

#ifdef WITH_CUDA
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include "gpu/cuda/Compositor_kernels.h"
#include "gpu/cuda/RayCaster.hpp"
#include "gpu/cuda/RadRenderer.hpp"
#else
#include "gpu/opencl/Compositor.hpp"
#include "gpu/opencl/Merger.hpp"
#include "gpu/opencl/RayCaster.hpp"
#include "gpu/opencl/RadRenderer.hpp"
#endif

#include "Filter.hpp"


using namespace std;

namespace xromm { namespace gpu
{

View::View(Camera& camera)
{
	camera_ = &camera;
	drr_enabled = true;
	rad_enabled = true;
	radRenderer_ = new RadRenderer();
	maxWidth_ = 2048;
	maxHeight_ = 2048;
	drrFilterBuffer_ = 0;
	radBuffer_ = 0;
	radFilterBuffer_ = 0;
	filterBuffer_ = 0;
	inited_ = false;
}

View::~View()
{
	for (int i = 0; i < drrRenderer_.size(); i++){
		delete drrRenderer_[i];
	}
	drrBuffer_.clear();
    delete radRenderer_;

    std::vector<Filter*>::iterator iter;
    for (iter = drrFilters_.begin(); iter != drrFilters_.end(); ++iter) {
        delete *iter;
    }

    for (iter = radFilters_.begin(); iter != radFilters_.end(); ++iter) {
        delete *iter;
    }
#ifdef WITH_CUDA
	cutilSafeCall(cudaFree(filterBuffer_));
    cutilSafeCall(cudaFree(drrBuffer_));
    cutilSafeCall(cudaFree(drrFilterBuffer_));
    cutilSafeCall(cudaFree(radBuffer_));
    cutilSafeCall(cudaFree(radFilterBuffer_));
#else
    delete filterBuffer_;
	for (int i = 0; i < drrBuffer_.size(); i++){
		delete drrBuffer_[i];
	}
	drrBuffer_.clear();
	delete drrBufferMerged_;
	delete drrFilterBuffer_;
    delete drrFilterBuffer_;
    delete radBuffer_;
    delete radFilterBuffer_;
#endif
}

void View::addDrrRenderer(){
	drrRenderer_.push_back(new RayCaster());
	drrBuffer_.push_back(0);
}

void
View::renderRad(Buffer* buffer, unsigned width, unsigned height)
{
#ifdef WITH_CUDA
	init();

    if (width > maxWidth_ || height > maxHeight_) {
        cerr << "View::renderRad(): ERROR: Buffer too large." << endl;
    }
    if (width > maxWidth_) {
        width = maxWidth_;
    }
    if (height > maxHeight_) {
        height = maxHeight_;
    }

    radRenderer_->render(radBuffer_, width, height);
    filter(radFilters_, radBuffer_, buffer, width, height);
#else
	init(width, height);
    radRenderer_->render(radBuffer_, width, height);
    filter(radFilters_, radBuffer_, buffer, width, height);
#endif
}

void
View::renderRad(unsigned int pbo, unsigned width, unsigned height)
{
#ifdef WITH_CUDA
	struct cudaGraphicsResource* pboCudaResource;
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&pboCudaResource, pbo,
        cudaGraphicsMapFlagsWriteDiscard));

    float* buffer = NULL;
    size_t numOfBytes;
    cutilSafeCall(cudaGraphicsMapResources(1, &pboCudaResource, 0));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&buffer,
                                                       &numOfBytes,
                                                       pboCudaResource));

    renderRad(radFilterBuffer_, width, height);
    gpu::composite(radFilterBuffer_,
                    radFilterBuffer_,
                    buffer,
                    width,
                    height);

    cutilSafeCall(cudaGraphicsUnmapResources(1, &pboCudaResource, 0));
    cutilSafeCall(cudaGraphicsUnregisterResource(pboCudaResource));
#else
	GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

	init(width, height);
    renderRad(radFilterBuffer_, width, height);
    composite(radFilterBuffer_, radFilterBuffer_, buffer, width, height);

	delete buffer;
#endif
}

void
View::renderDrr(Buffer* buffer, unsigned width, unsigned height)
{
#ifdef WITH_CUDA
	init();

    if (width > maxWidth_ || height > maxHeight_) {
        cerr << "View::renderDrr(): ERROR: Buffer too large." << endl;
    }
    if (width > maxWidth_) {
        width = maxWidth_;
    }
    if (height > maxHeight_) {
        height = maxHeight_;
    }

    drrRenderer_->render(drrBuffer_, width, height);
    filter(drrFilters_, drrBuffer_, buffer, width, height);
#else
	init(width, height);
	drrBufferMerged_->fill(0x00);
	for (int i = 0; i < drrRenderer_.size(); i++){
		drrRenderer_[i]->render(drrBuffer_[i], width, height);
		merge(drrBufferMerged_, drrBuffer_[i], drrBufferMerged_, width, height);
	}
	filter(drrFilters_, drrBufferMerged_, buffer, width, height);
#endif
}

void
View::renderDrr(unsigned int pbo, unsigned width, unsigned height)
{
#ifdef WITH_CUDA
	struct cudaGraphicsResource* pboCudaResource;
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&pboCudaResource, pbo,
        cudaGraphicsMapFlagsWriteDiscard));

    float* buffer = NULL;
    size_t numOfBytes;
    cutilSafeCall(cudaGraphicsMapResources(1, &pboCudaResource, 0));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&buffer,
                                                       &numOfBytes,
                                                       pboCudaResource));

    renderDrr(drrFilterBuffer_, width, height);
    gpu::composite(drrFilterBuffer_,
                    drrFilterBuffer_,
                    buffer,
                    width,
                    height);

    cutilSafeCall(cudaGraphicsUnmapResources(1, &pboCudaResource, 0));
    cutilSafeCall(cudaGraphicsUnregisterResource(pboCudaResource));
#else
	GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

	init(width, height);
    renderDrr(drrFilterBuffer_, width, height);
    composite(drrFilterBuffer_, drrFilterBuffer_, buffer, width, height);

	delete buffer;
#endif
}

void View::saveImage(std::string filename, int width, int height)
{
	fprintf(stderr, "Write to %s with size %d %d\n", filename.c_str(), width, height);

	Buffer* buffer = new Buffer(maxWidth_*maxHeight_*sizeof(float));
#ifdef WITH_CUDA
	init();

	if (width > maxWidth_ || height > maxHeight_) {
		cerr << "View::renderDrr(): ERROR: Buffer too large." << endl;
	}
	if (width > maxWidth_) {
		width = maxWidth_;
	}
	if (height > maxHeight_) {
		height = maxHeight_;
	}

	drrRenderer_->render(drrBuffer_, width, height);
	filter(drrFilters_, drrBuffer_, buffer, width, height);
#else
	init(width, height);

	drrBufferMerged_->fill(0x00);
	for(int i = 0; i < drrRenderer_.size(); i++){
		drrRenderer_[i]->render(drrBuffer_[i], width, height);
		merge(drrBufferMerged_, drrBuffer_[i], drrBufferMerged_, width, height);
	}
	filter(drrFilters_, drrBufferMerged_, buffer, width, height);

#endif
	float* host_image = new float[width*height];
	unsigned char* uchar_image = new unsigned char[width*height];

	// Copy the image to the host
	buffer->write(host_image, width*height*sizeof(float));
	//cudaMemcpy(host_image,dev_image,width*height*sizeof(float),cudaMemcpyDeviceToHost);

	// Copy to a char array
	/*for (int i = 0; i < width*height; i++) {
	uchar_image[i] = (int)(255*(1.0 - host_image[i]));
	}*/
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			uchar_image[y*width + x] = (int)(255 * (1.0 - host_image[(height - y - 1)*width + x]));
		}
	}
	ofstream file(filename.c_str(), ios::out);
	file << "P2" << endl;
	file << width << " " << height << endl;
	file << 255 << endl;
	for (int i = 0; i < width*height; i++) {
		file << (int)uchar_image[i] << " ";
	}

	delete[] uchar_image;
	delete[] host_image;
	delete buffer;
}

void
View::render(GLBuffer* buffer, unsigned width, unsigned height)
{
#ifdef WITH_CUDA
	init();

    if (drr_enabled) {
        renderDrr(drrFilterBuffer_, width, height);
    }
    else {
        cudaMemset(drrFilterBuffer_,0,width*height*sizeof(float));
    }

    if (rad_enabled) {
        renderRad(radFilterBuffer_, width, height);
    }
    else {
        cudaMemset(radFilterBuffer_,0,width*height*sizeof(float));
    }

    gpu::composite(drrFilterBuffer_,
                    radFilterBuffer_,
                    buffer,
                    width,
                    height);
#else
	init(width, height);

    if (drr_enabled) {
        renderDrr(drrFilterBuffer_, width, height);
    }
    else {
		drrFilterBuffer_->fill(0x00);
    }

    if (rad_enabled) {
        renderRad(radFilterBuffer_, width, height);
    }
    else {
		radFilterBuffer_->fill(0x00);
    }

    composite(drrFilterBuffer_, radFilterBuffer_, buffer, width, height);
#endif
}

void
View::render(unsigned int pbo, unsigned width, unsigned height)
{
#ifdef WITH_CUDA
	struct cudaGraphicsResource* pboCudaResource;
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&pboCudaResource, pbo,
        cudaGraphicsMapFlagsWriteDiscard));

    float* buffer = NULL;
    size_t numOfBytes;
    cutilSafeCall(cudaGraphicsMapResources(1, &pboCudaResource, 0));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&buffer,
                                                       &numOfBytes,
						                               pboCudaResource));

    render(buffer, width, height);

    cutilSafeCall(cudaGraphicsUnmapResources(1, &pboCudaResource, 0));
	cutilSafeCall(cudaGraphicsUnregisterResource(pboCudaResource));
#else
	GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

	init(width, height);
    render(buffer, width, height);

	delete buffer;
#endif
}


#ifdef WITH_CUDA
void
View::init()
{
    if (!filterBuffer_) {
        cutilSafeCall(cudaMalloc((void**)&filterBuffer_,
                                 maxWidth_*maxHeight_*sizeof(float)));
        cutilSafeCall(cudaMalloc((void**)&drrBuffer_,
                                 maxWidth_*maxHeight_*sizeof(float)));
        cutilSafeCall(cudaMalloc((void**)&drrFilterBuffer_,
                                 maxWidth_*maxHeight_*sizeof(float)));
        cutilSafeCall(cudaMalloc((void**)&radBuffer_,
                                 maxWidth_*maxHeight_*sizeof(float)));
        cutilSafeCall(cudaMalloc((void**)&radFilterBuffer_,
                                 maxWidth_*maxHeight_*sizeof(float)));
    }
}
#else
void
View::init(unsigned width, unsigned height)
{
    if (width > maxWidth_ || height > maxHeight_) {
		throw runtime_error("View::renderDrr(): Buffer too large");
    }

    if (!inited_) {
        filterBuffer_    = new Buffer(maxWidth_*maxHeight_*sizeof(float));
		for (int i = 0; i < drrBuffer_.size(); i++){
			drrBuffer_[i] = new Buffer(maxWidth_*maxHeight_*sizeof(float));
		}
		drrBufferMerged_ = new Buffer(maxWidth_*maxHeight_*sizeof(float));
		drrFilterBuffer_ = new Buffer(maxWidth_*maxHeight_*sizeof(float));
        radBuffer_       = new Buffer(maxWidth_*maxHeight_*sizeof(float));
        radFilterBuffer_ = new Buffer(maxWidth_*maxHeight_*sizeof(float));
		inited_ = true;
    }
}
#endif

void
View::filter(const std::vector<Filter*>& filters,
             const Buffer* input,
             Buffer* output,
             unsigned width,
             unsigned height)
{

#ifdef WITH_CUDA
	// If there are no filters simply copy the input to the output
    if (filters.size() == 0) {
        cudaMemcpy(output,
                   input,
                   width*height*sizeof(float),
                   cudaMemcpyDeviceToDevice);
        return;
    }

    // Determine which buffer will be used first so that the final
    // filter will place the results into output.
    float* buffer1;
    float* buffer2;
    if (filters.size()%2) {
        buffer1 = output;
        buffer2 = filterBuffer_;
    }
    else {
        buffer1 = filterBuffer_;
        buffer2 = output;
    }

    // Explicitly apply the first filter and altername buffers after
    vector<Filter*>::const_iterator iter = filters.begin();;

    if ((*iter)->enabled()) {
        (*iter)->apply(input, buffer1, (int)width, (int)height);
    }
    else {
        cudaMemcpy(buffer1,
                   input,
                   width*height*sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    for (iter += 1; iter != filters.end(); ++iter) {
        if ((*iter)->enabled()) {
            (*iter)->apply(buffer1, buffer2, (int)width, (int)height);
        }
        else {
            cudaMemcpy(buffer2,
                       buffer1,
                       width*height*sizeof(float),
                       cudaMemcpyDeviceToDevice);
        }
        swap(buffer1, buffer2);
    }
#else
    // If there are no filters simply copy the input to the output
    if (filters.size() == 0) {
		input->copy(output, width*height*sizeof(float));
        return;
    }

    // Determine which buffer will be used first so that the final
    // filter will place the results into output.
    Buffer* buffer1;
    Buffer* buffer2;
    if (filters.size()%2) {
        buffer1 = output;
        buffer2 = filterBuffer_;
    }
    else {
        buffer1 = filterBuffer_;
        buffer2 = output;
    }

    // Explicitly apply the first filter and altername buffers after
    vector<Filter*>::const_iterator iter = filters.begin();;

    if ((*iter)->enabled()) {
        (*iter)->apply(input, buffer1, (int)width, (int)height);
    }
    else {
		input->copy(buffer1, width*height*sizeof(float));
    }

    for (iter += 1; iter != filters.end(); ++iter) {
        if ((*iter)->enabled()) {
            (*iter)->apply(buffer1, buffer2, (int)width, (int)height);
        }
        else {
			buffer1->copy(buffer2, width*height*sizeof(float));
        }
        swap(buffer1, buffer2);
    }
#endif
}

} } // namespace xromm::opencl

