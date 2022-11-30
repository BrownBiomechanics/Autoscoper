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

/// \file TiffImage.h
/// \author Andy Loomis, Mark Howison

// Simple interface for reading and writing multiformat TIFF images using the
// libtiff library.

#ifndef XROMM_TIFF_IMAGE_H
#define XROMM_TIFF_IMAGE_H

#include <tiffio.h>

// This struct represents a generic TIFF image used for reading and writing.
// The raster data is stored as a void* and it is up to the application to
// interperet this as the appropriate type.
struct TiffImage
{
    // Baseline Tiff metadata
    uint32_t width;
    uint32_t height;
    uint16_t bitsPerSample;
    uint16_t photometric;
    uint16_t orientation;
    uint16_t samplesPerPixel;
    uint16_t planarConfig;
    uint16_t compression;

    // Extended Tiff metadata
    uint16_t sampleFormat;

    // Image data
    tdata_t data;
  size_t dataSize;
};

// This function reads in the metadata associated with a particular TIFF
// image. It returns 1 on success and 0 on failure.
int tiffImageReadMeta(TIFF* tif, TiffImage* img);

// This function dumps a summary of the metadata to stdout.
int tiffImageDumpMeta(TiffImage* img);

// This function reads in the metadata and image data associated with a
// particular TIFF image. It returns 1 on success and 0 on failure.
int tiffImageRead(TIFF* tif, TiffImage* img);

// Copy constructor duplicates a TiffImage struct and allocates and copies
// a new data region.
TiffImage* tiffImageCopy(TiffImage* img);

// Frees any memory associated with the TiffImage which was allocated during
// a call to tiffImageRead.
void tiffImageFree(TiffImage* img);

int tiffImageWrite(TIFF* tif, TiffImage* img);

// This function writes out an image to the current directory of an open
// TIFF file. The data is written in strips of the specified size. If the
// rowsPerStrip parameter is left at zero then a default value will be chosen
// by the tiff library. It returns 1 on success and 0 on failure.
int tiffImageWriteStripped(TIFF* tif, TiffImage* img, uint32_t rowsPerStrip);

// This function writes out an image to the current directory of an open
// TIFF file. The data is written in tiles of the specified dimension. If the
// tileWidth or tileHeight is set to zero than default values will be chosen
// by the tiff library. It returns 1 on success and 0 on failure.
int tiffImageWriteTiled(TIFF* tif, TiffImage* img, uint32_t tileWidth,
                        uint32_t tileHeight);

#endif // TIFF_IMAGE_H

