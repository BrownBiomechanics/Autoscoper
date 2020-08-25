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

/// \file RayCaster.cpp
/// \author Andy Loomis, Benjamin Knorlein

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "RayCaster.hpp"
#include "RayCaster_kernels.h"
#include "VolumeDescription.hpp"

using namespace std;

namespace xromm { namespace gpu {

static int num_ray_casters = 0;

RayCaster::RayCaster() : volumeDescription_(0),
                         sampleDistance_(0.5f),
                         rayIntensity_(10.0f),
                         cutoff_(0.0f),
                         name_("")
{
    stringstream name_stream;
    name_stream << "DrrRenderer" << (++num_ray_casters);
    name_ = name_stream.str();

    viewport_[0] = -1.0f;
    viewport_[1] = -1.0f;
    viewport_[2] =  2.0f;
    viewport_[3] =  2.0f;
}

RayCaster::~RayCaster()
{
    num_ray_casters = 0;
}

void
RayCaster::setVolume(VolumeDescription& volume)
{
    volumeDescription_ = &volume;
}

void
RayCaster::setInvModelView(const double* invModelView)
{
    if (!volumeDescription_) {
        cerr << "RayCaster: ERROR: Unable to calculate matrix." << endl;
        exit(0);
    }

    const float* invScale = volumeDescription_->invScale();
    const float* invTrans = volumeDescription_->invTrans();

    invModelView_[0]  = invModelView[0]*invScale[0]+
                        invModelView[12]*invTrans[0];
    invModelView_[1]  = invModelView[1]*invScale[0]+
                        invModelView[13]*invTrans[0];
    invModelView_[2]  = invModelView[2]*invScale[0]+
                        invModelView[14]*invTrans[0];
    invModelView_[3]  = invModelView[3]*invScale[0]+
                        invModelView[15]*invTrans[0];
    invModelView_[4]  = invModelView[4]*invScale[1]+
                        invModelView[12]*invTrans[1];
    invModelView_[5]  = invModelView[5]*invScale[1]+
                        invModelView[13]*invTrans[1];
    invModelView_[6]  = invModelView[6]*invScale[1]+
                        invModelView[14]*invTrans[1];
    invModelView_[7]  = invModelView[7]*invScale[1]+
                        invModelView[15]*invTrans[1];
    invModelView_[8]  = invModelView[8]*invScale[2]+
                        invModelView[12]*invTrans[2];
    invModelView_[9]  = invModelView[9]*invScale[2]+
                        invModelView[13]*invTrans[2];
    invModelView_[10] = invModelView[10]*invScale[2]+
                        invModelView[14]*invTrans[2];
    invModelView_[11] = invModelView[11]*invScale[2]+
                        invModelView[15]*invTrans[2];
    invModelView_[12] = invModelView[12]; 
    invModelView_[13] = invModelView[13]; 
    invModelView_[14] = invModelView[14]; 
    invModelView_[15] = invModelView[15];
}

void
RayCaster::setViewport(float x, float y, float width, float height)
{
    viewport_[0] = x;
    viewport_[1] = y;
    viewport_[2] = width;
    viewport_[3] = height;
}

void
RayCaster::render(float* buffer, size_t width, size_t height)
{
    if (!volumeDescription_) {
        cerr << "RayCaster: WARNING: No volume loaded. " << endl;
        return;
    }
  
    //float aspectRatio = (float)width/(float)height;
    volume_bind_array(volumeDescription_->image());
    volume_viewport(viewport_[0], viewport_[1], viewport_[2], viewport_[3]);
    volume_render(buffer,
                  width,
                  height,
                  invModelView_, 
                  sampleDistance_,
                  rayIntensity_,
                  cutoff_);
}

/*
template <class T>
bool
RayCaster::load(const Volume<T>& volume)
{
    // Crop the volume
    int min[3] = { volume.width(), volume.height(), volume.depth() };
    int max[3] = { 0 }; 
    const T* dp1 = volume.data();
    for (int k = 0; k < volume.depth(); k++) {
        bool nonZeroCol = false;
        for (int i = 0; i < volume.height(); i++) {
            bool nonZeroRow = false;
            for (int j = 0; j < volume.width(); j++) {
                if (*dp1++ != T(0)) {
                    if (j < min[0]) {
                        min[0] = j;
                    }
                    if (j > max[0]) {
                        max[0] = j;
                    }
                    nonZeroRow = true;
                }
            }
            if (nonZeroRow) {
                if (i < min[1]) {
                    min[1] = i;
                }
                if (i > max[1]) {
                    max[1] = i;
                }
                nonZeroCol = true;
            }
        }
        if (nonZeroCol) {
            if (k < min[2]) {
                min[2] = k;
            }
            if (k > max[2]) {
                max[2] = k;
            }
        }
    }

    // The volume is empty
    if (min[0] > max[0] || min[1] > max[1] || min[2] > max[2]) { 
        std::cerr << "Empty Volume" << std::endl;
        return false;
    }

    // Copy to the cropped volume
    int dim[3] = { max[0]-min[0]+1, max[1]-min[1]+1, max[2]-min[2]+1 };
    T* data = new T[dim[0]*dim[1]*dim[2]];
    T* dp2 = data; 
    for (int k = min[2]; k < max[2]+1; k++) {
        for (int i = min[1]; i < max[1]+1; i++) {
            for (int j = min[0]; j < max[0]+1; j++) {
                *dp2++ = volume.data()[k*volume.width()*volume.height()+
                                       i*volume.width()+j];            
            }
        }
    }
  
    // Calculate the offset and size of the sub-volume
    invScale_[0] = 1.0f/(float)(volume.scaleX()*dim[0]);
    invScale_[1] = 1.0f/(float)(volume.scaleY()*dim[1]);
    invScale_[2] = 1.0f/(float)(volume.scaleZ()*dim[2]);

    invTrans_[0] = -min[0]/(float)dim[0];
    invTrans_[1] = -((volume.height()-max[1]-1)/(float)dim[1]);
    invTrans_[2] = min[2]/(float)dim[2];

    // Free any previously allocated memory.
    cutilSafeCall(cudaFreeArray(array_));
    
    // Create a 3D array.
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
    cudaExtent extent = make_cudaExtent(dim[0], dim[1], dim[2]);
    cutilSafeCall(cudaMalloc3DArray(&array_, &desc, extent));

    // Copy volume to 3D array.
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(data),
                                            extent.width*sizeof(T),
                                            extent.width, extent.height);
    copyParams.dstArray = array_;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cutilSafeCall(cudaMemcpy3D(&copyParams));  

    return true;
}
*/

} } // namespace xromm::cuda

