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

/// \file Volume.cpp
/// \author Andy Loomis

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Volume.hpp"
#include "TiffImage.h"

using namespace std;

namespace xromm
{

Volume::Volume(const string& filename)
    : name_(filename),
      width_(0), height_(0), depth_(0), bps_(0),
      scaleX_(1.0f), scaleY_(1.0f), scaleZ_(1.0f),
      flipX_(false), flipY_(false), flipZ_(false),
      data_(0)
{
    TIFFSetWarningHandler(0);
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (!tif) {
        throw runtime_error("Unable to open volume file: " + filename);
    }

    // Determine the size and format of each slice
    TiffImage img;
    tiffImageReadMeta(tif, &img);

    if (img.samplesPerPixel != 1 || img.sampleFormat != 1) {
        throw runtime_error("Unsupported image format");
    }

    // Count the number of slices
    int dircount = 0;
    do {
        dircount++;
    } while(TIFFReadDirectory(tif));

    width_ = img.width;
    height_ = img.height;
    depth_ = dircount;
    bps_ = img.bitsPerSample;
    data_ = new unsigned char[width_*height_*depth_*(bps_/8)];

    unsigned char* dp = (unsigned char*)data_;
    for (size_t i = 0; i < depth_; ++i) {
        TIFFSetDirectory(tif, i);
        tiffImageRead(tif, &img);

        if (img.width != width_ ||
            img.height != height_ ||
            img.bitsPerSample != bps_) {
            throw runtime_error("Non uniform volume slices.");
        }

        memcpy(dp, img.data, width_*height_*(bps_/8));
        tiffImageFree(&img);
        dp += width_*height_*(bps_/8);
    }

    TIFFClose(tif);
}


Volume::Volume(const Volume& volume) : name_(volume.name_),
                                       width_(volume.width_),
                                       height_(volume.height_),
                                       depth_(volume.depth_),
                                       bps_(volume.bps_),
                                       scaleX_(volume.scaleX_),
                                       scaleY_(volume.scaleY_),
                                       scaleZ_(volume.scaleZ_),
                                       flipX_(volume.flipX_),
                                       flipY_(volume.flipY_),
                                       flipZ_(volume.flipZ_),
                                       data_(0)
{
    size_t size = width_*height_*depth_*(bps_/8);
    data_ = new char[size];
    memcpy(data_, volume.data_, size);
}

Volume::~Volume()
{
    delete[] reinterpret_cast<char*>(data_);
}

Volume&
Volume::operator=(const Volume& volume)
{
    name_ = volume.name_;
    width_ = volume.width_;
    height_ = volume.height_;
    depth_ = volume.depth_;
    bps_ = volume.bps_;
    scaleX_ = volume.scaleX_;
    scaleY_ = volume.scaleY_;
    scaleZ_ = volume.scaleZ_;
    flipX_ = volume.flipX_;
    flipY_ = volume.flipY_;
    flipZ_ = volume.flipZ_;

    delete[] reinterpret_cast<char*>(data_);
    size_t size = width_*height_*depth_*(bps_/8);
    data_ = new char[size];
    memcpy(data_, volume.data_, size);

    return *this;
}

} // namespace xromm
