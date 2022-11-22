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

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include "gpu/cuda/Compositor_kernels.h"
#include "gpu/cuda/RayCaster.hpp"
#include "gpu/cuda/RadRenderer.hpp"
#include "gpu/cuda/Merger_kernels.h"
#include "gpu/cuda/BackgroundRenderer.hpp"
#include "gpu/cuda/DRRBackground_kernels.h"

#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
#include "gpu/opencl/Compositor.hpp"
#include "gpu/opencl/Merger.hpp"
#include "gpu/opencl/RayCaster.hpp"
#include "gpu/opencl/RadRenderer.hpp"
#include "gpu/opencl/BackgroundRenderer.hpp"
#include "gpu/opencl/DRRBackground.hpp"
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
  backgroundRenderer_ = new BackgroundRenderer();
  maxWidth_ = 3000;
  maxHeight_ = 3000;
  drrFilterBuffer_ = 0;
  radBuffer_ = 0;
  radFilterBuffer_ = 0;
  filterBuffer_ = 0;
  backgroundThreshold_ = -1.0f;
  inited_ = false;
}

View::~View()
{
    std::vector<Filter*>::iterator iter;
    for (iter = drrFilters_.begin(); iter != drrFilters_.end(); ++iter) {
        delete *iter;
    }

    for (iter = radFilters_.begin(); iter != radFilters_.end(); ++iter) {
        delete *iter;
    }
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  cutilSafeCall(cudaFree(filterBuffer_));
  for(int i = 0; i < drrBuffer_.size(); i++){
    cutilSafeCall(cudaFree(drrBuffer_[i]));
  }
  cutilSafeCall(cudaFree(drrBufferMerged_));
    cutilSafeCall(cudaFree(drrFilterBuffer_));
    cutilSafeCall(cudaFree(radBuffer_));
    cutilSafeCall(cudaFree(radFilterBuffer_));
  cutilSafeCall(cudaFree(backgroundmask_));
  cutilSafeCall(cudaFree(drr_mask_));

#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
    delete filterBuffer_;
  for (int i = 0; i < drrBuffer_.size(); i++){
    delete drrBuffer_[i];
  }
  delete drrBufferMerged_;
  delete drrFilterBuffer_;
    delete radBuffer_;
    delete radFilterBuffer_;

  delete backgroundmask_;
  delete drr_mask_;
#endif
  drrBuffer_.clear();

  for (int i = 0; i < drrRenderer_.size(); i++) {
    delete drrRenderer_[i];
  }
  delete radRenderer_;
  delete backgroundRenderer_;
}

void View::addDrrRenderer(){
  drrRenderer_.push_back(new RayCaster());
  drrBuffer_.push_back(0);
}

void
View::renderRad(Buffer* buffer, unsigned width, unsigned height)
{
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  init(width, height);
    radRenderer_->render(radBuffer_, width, height);
    filter(radFilters_, radBuffer_, buffer, width, height);
#endif
}

void
View::renderBackground(Buffer* buffer, unsigned width, unsigned height)
{
  if (backgroundThreshold_ < 0.0)
    return;

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  init();

  if (width > maxWidth_ || height > maxHeight_) {
    cerr << "View::renderBackground(): ERROR: Buffer too large." << endl;
  }
  if (width > maxWidth_) {
    width = maxWidth_;
  }
  if (height > maxHeight_) {
    height = maxHeight_;
  }

  backgroundRenderer_->render(buffer, width, height, backgroundThreshold_);

#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  backgroundRenderer_->render(buffer, width, height, backgroundThreshold_);
#endif
}

void View::renderDRRMask(Buffer* in_buffer, Buffer* out_buffer, unsigned width, unsigned height)
{
  drr_background(in_buffer, out_buffer, width, height);
}

void
View::renderDrr(Buffer* buffer, unsigned width, unsigned height)
{
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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

  cudaMemset(drrBufferMerged_, 0, width*height*sizeof(float));
  for (int i = 0; i < drrRenderer_.size(); i++){
    drrRenderer_[i]->render(drrBuffer_[i], width, height);
    gpu::merge(drrBufferMerged_, drrBuffer_[i], drrBufferMerged_, width, height);
  }
  filter(drrFilters_, drrBufferMerged_, buffer, width, height);
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  init(width, height);
  drrBufferMerged_->fill((char) 0x00);
  for (int i = 0; i < drrRenderer_.size(); i++){
    drrRenderer_[i]->render(drrBuffer_[i], width, height);
    merge(drrBufferMerged_, drrBuffer_[i], drrBufferMerged_, width, height);
  }
  filter(drrFilters_, drrBufferMerged_, buffer, width, height);
#endif
}

void View::renderDrrSingle(int volume, Buffer* buffer, unsigned width, unsigned height)
{
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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

  drrRenderer_[volume]->render(drrBuffer_[volume], width, height);
  filter(drrFilters_, drrBuffer_[volume], buffer, width, height);

#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  init(width, height);
  drrRenderer_[volume]->render(drrBuffer_[volume], width, height);
  filter(drrFilters_, drrBuffer_[volume], buffer, width, height);
#endif
}

void
View::renderDrr(unsigned int pbo, unsigned width, unsigned height)
{
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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

  gpu::fill(backgroundmask_, maxWidth_*maxHeight_, 1.0f);
  gpu::fill(drr_mask_, maxWidth_*maxHeight_, 1.0f);

    gpu::composite(drrFilterBuffer_,
                    drrFilterBuffer_,
          backgroundmask_,
          drr_mask_,
                    buffer,
                    width,
                    height);

    cutilSafeCall(cudaGraphicsUnmapResources(1, &pboCudaResource, 0));
    cutilSafeCall(cudaGraphicsUnregisterResource(pboCudaResource));
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

  init(width, height);
    renderDrr(drrFilterBuffer_, width, height);
  backgroundmask_->fill(1.0f);
  drr_mask_->fill(1.0f);
  composite(drrFilterBuffer_, drrFilterBuffer_, backgroundmask_, drr_mask_, buffer, width, height);

  delete buffer;
#endif
}

void View::saveImage(std::string filename, int width, int height)
{
  fprintf(stderr, "Write to %s with size %d %d\n", filename.c_str(), width, height);

  Buffer* buffer = new Buffer(maxWidth_*maxHeight_*sizeof(float));
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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

  cudaMemset(drrBufferMerged_, 0, width*height*sizeof(float));
  for (int i = 0; i < drrRenderer_.size(); i++){
    drrRenderer_[i]->render(drrBuffer_[i], width, height);
    gpu::merge(drrBufferMerged_, drrBuffer_[i], drrBufferMerged_, width, height);
  }
  filter(drrFilters_, drrBufferMerged_, buffer, width, height);

#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  init(width, height);

  drrBufferMerged_->fill((char)0x00);
  for(int i = 0; i < drrRenderer_.size(); i++){
    drrRenderer_[i]->render(drrBuffer_[i], width, height);
    merge(drrBufferMerged_, drrBuffer_[i], drrBufferMerged_, width, height);
  }
  filter(drrFilters_, drrBufferMerged_, buffer, width, height);

#endif
  float* host_image = new float[width*height];
  unsigned char* uchar_image = new unsigned char[width*height];

  // Copy the image to the host
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
  cudaMemcpy(host_image, buffer, width*height*sizeof(float), cudaMemcpyDeviceToHost);
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  buffer->write(host_image, width*height*sizeof(float));
#endif

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
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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

  renderBackground(backgroundmask_, width, height);
  renderDRRMask(drrFilterBuffer_, drr_mask_, width, height);

    gpu::composite(drrFilterBuffer_,
                    radFilterBuffer_,
          backgroundmask_,
          drr_mask_,
                    buffer,
                    width,
                    height);
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  init(width, height);
  const char c = 0x00;
    if (drr_enabled) {
        renderDrr(drrFilterBuffer_, width, height);
    }
    else {
    drrFilterBuffer_->fill(c);
    }

    if (rad_enabled) {
        renderRad(radFilterBuffer_, width, height);
    }
    else {
    radFilterBuffer_->fill(c);
    }
  renderBackground(backgroundmask_, width, height);
  renderDRRMask(drrFilterBuffer_, drr_mask_, width, height);
  composite(drrFilterBuffer_, radFilterBuffer_, backgroundmask_, drr_mask_, buffer, width, height);
#endif
}

void
View::render(unsigned int pbo, unsigned width, unsigned height)
{
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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

#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

  init(width, height);
    render(buffer, width, height);

  delete buffer;
#endif
}


void View::updateBackground(const float* buffer, unsigned width, unsigned height)
{
  backgroundRenderer_->set_back(buffer,width,height);
}


#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
void
View::init()
{
  if (!filterBuffer_) {
    cutilSafeCall(cudaMalloc((void**)&filterBuffer_, maxWidth_ * maxHeight_ * sizeof(float)));
    for (int i = 0; i < drrBuffer_.size(); i++)
      cutilSafeCall(cudaMalloc((void**)&drrBuffer_[i], maxWidth_ * maxHeight_ * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&drrBufferMerged_, maxWidth_ * maxHeight_ * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&drrFilterBuffer_, maxWidth_ * maxHeight_ * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&radBuffer_, maxWidth_ * maxHeight_ * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&radFilterBuffer_, maxWidth_ * maxHeight_ * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&backgroundmask_, maxWidth_ * maxHeight_ * sizeof(float)));
    cutilSafeCall(cudaMalloc((void**)&drr_mask_, maxWidth_ * maxHeight_ * sizeof(float)));
    // fill the array
    gpu::fill(backgroundmask_, maxWidth_ * maxHeight_, 1.0f);
    gpu::fill(drr_mask_, maxWidth_ * maxHeight_, 1.0f);
  }
}
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
void
View::init(unsigned width, unsigned height)
{
  if (width*height > maxWidth_ * maxHeight_) {
    throw runtime_error("GPU Buffers too small");
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
    backgroundmask_  = new Buffer(maxWidth_*maxHeight_*sizeof(float));
    drr_mask_    = new Buffer(maxWidth_*maxHeight_*sizeof(float));
    backgroundmask_->fill(1.0f);
    drr_mask_->fill(1.0f);
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

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
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
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
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

