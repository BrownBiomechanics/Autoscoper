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

/// \file VolumeDescription.cpp
/// \author Andy Loomis, Mark Howison



#include <iostream>
#include <limits>
#include <fstream>
#include <vector>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#endif

#include "Volume.hpp"
#include "VolumeDescription.hpp"

#undef max
#undef min

using namespace std;

template <class T>
void flipVolume(const T* data,
				T* dest,
                int width,
                int height,
                int depth,
                bool flipX,
				bool flipY,
				bool flipZ)
{
	int x,y,z;
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                x = flipX ? (width-1) - j : j;
				y = flipY ? (height-1) - i : i;
				z = flipZ ? (depth-1) - k : k;

				dest[z*width*height+y*width+x]= data[k*width*height+i*width+j];
            }
        }
    }
}

template <class T>
void cropVolume(const T* data,
                int width,
                int height,
                int depth,
                int* min,
                int* max)
{
    min[0] = width;
    min[1] = height;
    min[2] = depth;
    max[0] = 0;
    max[1] = 0;
    max[2] = 0;
    const T* dp1 = data;
    for (int k = 0; k < depth; k++) {
        bool nonZeroCol = false;
        for (int i = 0; i < height; i++) {
            bool nonZeroRow = false;
            for (int j = 0; j < width; j++) {
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
}

template <class T>
void copyVolume(T* dest,
                const T* src,
                int width,
                int height,
                int depth,
                const int* min,
                const int* max,
                T* minVal,
                T* maxVal)
{
    *minVal = numeric_limits<T>::max();
    *maxVal = numeric_limits<T>::min();
    for (int k = min[2]; k < max[2]+1; k++) {
        for (int i = min[1]; i < max[1]+1; i++) {
            for (int j = min[0]; j < max[0]+1; j++) {
                if (src[k*width*height+i*width+j] < *minVal) {
                    *minVal = src[k*width*height+i*width+j];
                }
                if (src[k*width*height+i*width+j] > *maxVal) {
                    *maxVal = src[k*width*height+i*width+j];
                }
                *dest++ = src[k*width*height+i*width+j];
            }
        }
    }
}

namespace xromm { namespace gpu
{

VolumeDescription::VolumeDescription(const Volume& volume)
    : minValue_(0.0f), maxValue_(1.0f), image_(0)
{
    // Crop the volume
    int min[3], max[3];
	vector<char> data_flipped(volume.width()*volume.height()*volume.depth()*(volume.bps()/8));
    
    switch(volume.bps()) {
        case 8: {
			flipVolume((unsigned char*)volume.data(),
					(unsigned char*)&data_flipped[0],
                    (int)volume.width(),
                    (int)volume.height(),
                    (int)volume.depth(),
					(bool)volume.flipX(),
                    (bool)volume.flipY(),
                    (bool)volume.flipZ()
			);

            cropVolume((unsigned char*)&data_flipped[0],
                    (int)volume.width(),
                    (int)volume.height(),
                    (int)volume.depth(),
                    min,
                    max);
            break;
        }
        case 16: {
			flipVolume((unsigned short*)volume.data(),
					(unsigned short*)&data_flipped[0],
                    (int)volume.width(),
                    (int)volume.height(),
                    (int)volume.depth(),
					(bool)volume.flipX(),
                    (bool)volume.flipY(),
                    (bool)volume.flipZ()
			);
            cropVolume((unsigned short*)&data_flipped[0],
                    (int)volume.width(),
                    (int)volume.height(),
                    (int)volume.depth(),
                    min,
                    max);
            break;
        }
        default: {
            cerr << "VolumeDescription(): Unsupported bit-depth "
                                          << volume.bps() << endl;
            exit(0);
        }
    }

    // The volume is empty
    if (min[0] > max[0] || min[1] > max[1] || min[2] > max[2]) {
        std::cerr << "Empty Volume" << std::endl;
        exit(0);
    }

    // Copy to the cropped volume
    int dim[3] = { max[0]-min[0]+1, max[1]-min[1]+1, max[2]-min[2]+1 };
    vector<char> data(dim[0]*dim[1]*dim[2]*(volume.bps()/8));
    switch(volume.bps()) {
        case 8: {
            unsigned char minVal, maxVal;
            copyVolume((unsigned char*)&data[0],
                    (unsigned char*)&data_flipped[0],
                    (int)volume.width(),
                    (int)volume.height(),
                    (int)volume.depth(),
                    min,
                    max,
                    &minVal,
                    &maxVal);
            minValue_ = minVal/(float)numeric_limits<unsigned char>::max();
            maxValue_ = maxVal/(float)numeric_limits<unsigned char>::max();
            break;
        }
        case 16: {
            unsigned short minVal, maxVal;
            copyVolume((unsigned short*)&data[0],
                    (unsigned short*)&data_flipped[0],
                    (int)volume.width(),
                    (int)volume.height(),
                    (int)volume.depth(),
                    min,
                    max,
                    &minVal,
                    &maxVal);
            minValue_ = minVal/(float)numeric_limits<unsigned short>::max();
            maxValue_ = maxVal/(float)numeric_limits<unsigned short>::max();
            break;
        }
        default:
            cerr << "VolumeDescription(): Unsupported bit-depth "
                                          << volume.bps() << endl;
            exit(0);
    }

    // Calculate the offset and size of the sub-volume
    invScale_[0] = 1.0f/(float)(volume.scaleX()*dim[0]);
    invScale_[1] = 1.0f/(float)(volume.scaleY()*dim[1]);
    invScale_[2] = 1.0f/(float)(volume.scaleZ()*dim[2]);

    invTrans_[0] = -min[0]/(float)dim[0];
    invTrans_[1] = -((volume.height()-max[1]-1)/(float)dim[1]);
    invTrans_[2] = min[2]/(float)dim[2];
    // Free any previously allocated memory.
	
#ifdef WITH_CUDA
	// Free any previously allocated memory.
    cutilSafeCall(cudaFreeArray(image_));

    // Create a 3D array.
    cudaChannelFormatDesc desc;
    switch(volume.bps()) {
        case 8: desc = cudaCreateChannelDesc<unsigned char>(); break;
        case 16: desc = cudaCreateChannelDesc<unsigned short>(); break;
        default:
                 cerr << "VolumeDescription(): Unsupported bit-depth "
                                               << volume.bps() << endl;
                 exit(0);
    }
    cudaExtent extent = make_cudaExtent(dim[0], dim[1], dim[2]);
    cutilSafeCall(cudaMalloc3DArray(&image_, &desc, extent));

    // Copy volume to 3D array.
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(&data[0],
            extent.width*(volume.bps()/8),
            extent.width, extent.height);
    copyParams.dstArray = image_;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cutilSafeCall(cudaMemcpy3D(&copyParams));
#else
	if (image_) delete image_;

    // Create a 3D array.
	cl_image_format format;
	format.image_channel_order = CL_R;
    switch (volume.bps()) {
        case 8:  format.image_channel_data_type = CL_UNORM_INT8; break;
        case 16: format.image_channel_data_type = CL_UNORM_INT16; break;
        default:
            cerr << "VolumeDescription(): unsupported bit depth "
                 << volume.bps() << endl;
            return;
    }

	size_t sdim[3] = { (size_t)dim[0], (size_t)dim[1], (size_t)dim[2] };
	image_ = new Image(sdim, &format, CL_MEM_READ_ONLY);
	image_->read(&data[0]);
#endif
}

VolumeDescription::~VolumeDescription()
{
#ifdef WITH_CUDA
	cutilSafeCall(cudaFreeArray(image_));
#else
	if (image_) delete image_;
#endif
}

} } // namespace xromm::opencl

