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

/// \file TiffImage.c
/// \author Andy Loomis
#include <cstring>
#include <cstdlib>
#include <iostream>

#include "TiffImage.h"

// Helper Functions

static int tiffReadContigTiledData(TIFF* tif, uint8_t* data, uint32_t width,
                                   uint32_t height, uint32_t spp);

static int tiffReadContigStrippedData(TIFF* tif, uint8_t* data, uint32_t width,
                                      uint32_t height, uint32_t spp);

static int tiffImageWriteMeta(TIFF* tif, TiffImage* img);

static int tiffWriteContigTiledData(TIFF* tif, uint8_t* data, uint32_t width,
                                    uint32_t height, uint32_t spp);

static int tiffWriteContigStrippedData(TIFF* tif, uint8_t* data, uint32_t width,
                                       uint32_t height, uint32_t spp);

// Function definitions

int tiffImageReadMeta(TIFF* tif, TiffImage* img)
{
    if (!tif) {
        printf("Invalid TIFF handle.\n");
        return 0;
    }

    if (!img) {
        printf("Invalid TiffImage pointer.\n");
        return 0;
    }

    // Read  baseline TIFF tags. The width, height, and photometric
    // interperetation are required tags. The remaining tags have
    // default values if they are not specified.
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &img->width) ||
        !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &img->height)) {
        return 0;
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &img->bitsPerSample);

    if (!TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &img->photometric)) {
        return 0;
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_ORIENTATION, &img->orientation);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &img->samplesPerPixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &img->planarConfig);
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &img->compression);

    // Read in extended TIFF tags.
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &img->sampleFormat);

    return 1;
}

int tiffImageDumpMeta(TiffImage* img)
{
    if (!img) {
        printf("Invalid TiffImage pointer.\n");
        return 0;
    }
    std::cout << "width: " << img->width << "\n";
    std::cout << "height: " << img->height << "\n";
    std::cout << "bitsPerSample: " << img->bitsPerSample << "\n";
    std::cout << "photometric: " << img->photometric << "\n";
    std::cout << "orientation: " << img->orientation << "\n";
    std::cout << "samplesPerPixel: " << img->samplesPerPixel << "\n";
    std::cout << "planarConfig: " << img->planarConfig << "\n";
    std::cout << "compression: " << img->compression << "\n";
    std::cout << "sampleFormat: " << img->sampleFormat << std::endl;
  return 1;
}

int tiffImageRead(TIFF* tif, TiffImage* img)
{
    if (!tiffImageReadMeta(tif, img)) {
        return 0;
    }

    // Allocate buffer for image data.
    img->dataSize = img->height*(size_t)TIFFRasterScanlineSize(tif);
    img->data = malloc(img->dataSize);
    if (!img->data) {
        printf("Unable to allocate space for image (%lu bytes).\n", img->dataSize);
        return 0;
    }

    if (img->planarConfig != PLANARCONFIG_CONTIG) {
        printf("Support for planar data has not yet been implemented.\n");
        free(img->data);
        return 0;
    }

    if (TIFFIsTiled(tif)) {
        return tiffReadContigTiledData(tif, (uint8_t*)img->data, img->width,
                                       img->height, img->samplesPerPixel);
    }
    else {
        return tiffReadContigStrippedData(tif, (uint8_t*)img->data, img->width,
                                          img->height, img->samplesPerPixel);
    }
}

TiffImage* tiffImageCopy(TiffImage* img)
{
  TiffImage* copy = (TiffImage*)malloc(sizeof(TiffImage));
  if (!copy) {
        printf("Unable to allocate space for TiffImage struct.\n");
        return 0;
    }

  memcpy(copy, img, sizeof(TiffImage));

    // Allocate buffer for image data.
    img->data = malloc(img->dataSize);
    if (!img->data) {
        printf("Unable to allocate space for image (%lu bytes).\n", img->dataSize);
        return 0;
    }

  memcpy(copy->data, img->data, img->dataSize);

  return copy;
}

void tiffImageFree(TiffImage* img)
{
    if (img) {
        free(img->data);
    }
}

int tiffImageWrite(TIFF* tif, TiffImage* img)
{
    return tiffImageWriteStripped(tif, img, 0);
}

int tiffImageWriteStripped(TIFF* tif, TiffImage* img, uint32_t rowsPerStrip)
{
    if (!tiffImageWriteMeta(tif, img)) {
        return 0;
    }

    // Write the rows per strip tag.
    rowsPerStrip = TIFFDefaultStripSize(tif, rowsPerStrip);
    if (!TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip)) {
        return 0;
    }

    switch (img->planarConfig) {
        case PLANARCONFIG_CONTIG:
            return tiffWriteContigStrippedData(tif, (uint8_t*)img->data,
                                               img->width, img->height,
                                               img->samplesPerPixel);
        case PLANARCONFIG_SEPARATE:
            printf("Support for planar data has not yet been implemented.\n");
            return 0;
        default:
            printf("Unknown planar configuration.\n");
            return 0;
    }
}

int tiffImageWriteTiled(TIFF* tif, TiffImage* img, uint32_t tileWidth,
                        uint32_t tileHeight)
{
    if (!tiffImageWriteMeta(tif, img)) {
        return 0;
    }

    // Determine the tile size.
    TIFFDefaultTileSize(tif, &tileWidth, &tileHeight);
    if (!TIFFSetField(tif, TIFFTAG_TILEWIDTH, tileWidth) ||
        !TIFFSetField(tif, TIFFTAG_TILELENGTH, tileHeight)) {
        return 0;
    }

    switch (img->planarConfig) {
        case PLANARCONFIG_CONTIG:
            return tiffWriteContigTiledData(tif, (uint8_t*)img->data, img->width,
                                            img->height, img->samplesPerPixel);
        case PLANARCONFIG_SEPARATE:
            printf("Support for planar data has not yet been implemented.\n");
            return 0;
        default:
            printf("Unknown planar configuration.\n");
            return 0;
    }
}

//////////////////////////////////////////////////////////////////////////////

static int tiffReadContigStrippedData(TIFF* tif, uint8_t* data, uint32_t width,
                                      uint32_t height, uint32_t spp)
{
    tsize_t scanlineSize = TIFFScanlineSize(tif);

    // Read in row data one line at a time.
    uint32_t y;
    for (y = 0; y < height; ++y) {

        // Read in the strip data.
        if (TIFFReadScanline(tif, (void*)data, y, 0) < 0) {
            return 0;
        }

        // Increment the image buffer pointer to the next row
        data += scanlineSize;
    }

    return 1;
}

static int tiffReadContigTiledData(TIFF* tif, uint8_t* data, uint32_t width,
                                   uint32_t height, uint32_t spp)
{
    // Get the tile dimensions.
    uint32_t tileWidth, tileHeight;
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);

    // Allocate space for tile buffer.
    uint8_t* tileBuf = (uint8_t*)_TIFFmalloc(TIFFTileSize(tif));
    if (!tileBuf) {
        printf("Unable to allocate space for tile (%d bytes).\n",
               TIFFTileSize(tif));
        return 0;
    }

    // Get the size in bytes of an image row and a tile row.
    tsize_t scanlineSize = TIFFScanlineSize(tif);
    tsize_t tileRowSize = TIFFTileRowSize(tif);

    uint32_t y;
    for (y = 0; y < height; y += tileHeight) {

        // Number of rows in these tiles that are visible
        uint32_t numRows = y+tileHeight > height? height-y: tileHeight;

        // Size of the offset from the first column of the image to the
        // first column of the tile.
        tsize_t byteOffset = 0;

        uint32_t x;
        for (x = 0; x < width; x += tileWidth) {

            // Read in the tile data.
            if (TIFFReadTile(tif, (tdata_t)tileBuf, x, y, 0, 0) < 0) {
                _TIFFfree(tileBuf);
                return 0;
            }

            // The size in bytes of the visible portion of each row
            tsize_t rowSize = byteOffset+tileRowSize > scanlineSize?
                              scanlineSize-byteOffset: tileRowSize;

            // Copy each tile into the image buffer one row at a time.
            uint32_t i;
            for (i = 0; i < numRows; ++i) {
                memcpy(data+i*scanlineSize+byteOffset,
                       tileBuf+i*tileRowSize,
                       rowSize);
            }

            // Increment the offset by the tile's width.
            byteOffset += tileRowSize;
        }

        // Increment the image buffer pointer for the next row of tiles
        data += tileHeight*scanlineSize;
    }

    // Free the tile buffer
    _TIFFfree(tileBuf);

    return 1;
}

static int tiffImageWriteMeta(TIFF* tif, TiffImage* img)
{
    if (TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img->width) &&
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img->height) &&
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, img->bitsPerSample) &&
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, img->photometric) &&
        TIFFSetField(tif, TIFFTAG_ORIENTATION, img->orientation) &&
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, img->samplesPerPixel) &&
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, img->planarConfig) &&
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, img->sampleFormat) &&
        TIFFSetField(tif, TIFFTAG_COMPRESSION, img->compression)) {
        return 1;
    }
    else {
        return 0;
    }
}

static int tiffWriteContigStrippedData(TIFF* tif, uint8_t* data, uint32_t width,
                                       uint32_t height, uint32_t spp)
{
    uint32_t rowsPerStrip;
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);

    tstrip_t strip = 0;

    // Write out one strip at a time.
    uint32_t y;
    for (y = 0; y < height; y += rowsPerStrip) {

        // The number of visible rows in this strip
        uint32_t numRows = y+rowsPerStrip > height? height-y: rowsPerStrip;

        // The number of bytes in this strip
        tsize_t stripSize = TIFFVStripSize(tif, numRows);

        // Write out the strip.
        if (TIFFWriteEncodedStrip(tif, strip++, data, stripSize) < 0) {
            return 0;
        }

        // Increment the data pointer for the next strip.
        data += stripSize;
    }

    return 1;
}

static int tiffWriteContigTiledData(TIFF* tif, uint8_t* data, uint32_t width,
                                    uint32_t height, uint32_t spp)
{
    // Get the tile dimensions.
    uint32_t tileWidth, tileHeight;
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);

    // Allocate space for tile buffer.
    uint8_t* tileBuf = (uint8_t*)_TIFFmalloc(TIFFTileSize(tif));
    if (!tileBuf) {
        printf("Unable to allocate space for tile (%d bytes).\n",
               TIFFTileSize(tif));
        return 0;
    }

    // Get the size in bytes of an image row and a tile row.
    tsize_t scanlineSize = TIFFScanlineSize(tif);
    tsize_t tileRowSize = TIFFTileRowSize(tif);

    uint32_t y;
    for (y = 0; y < height; y += tileHeight) {

        // Number of rows in these tiles that are visible
        uint32_t numRows = y+tileHeight > height? height-y: tileHeight;

        // Size of the offset from the first column of the image to the
        // first column of the tile.
        tsize_t byteOffset = 0;

        uint32_t x;
        for (x = 0; x < width; x += tileWidth) {

            // The size in bytes of the visible portion of each row
            tsize_t rowSize = byteOffset+tileRowSize > scanlineSize?
                              scanlineSize-byteOffset: tileRowSize;

            // Copy each tile row into the tile buffer one row at a time.
            uint32_t i;
            for (i = 0; i < numRows; ++i) {
                memcpy(tileBuf+i*tileRowSize,
                       data+i*scanlineSize+byteOffset,
                       rowSize);
            }

            // Write out the tile data.
            if (TIFFWriteTile(tif, (tdata_t)tileBuf, x, y, 0, 0) < 0) {
                _TIFFfree(tileBuf);
                return 0;
            }

            // Increment the offset by the tile's width.
            byteOffset += tileRowSize;
        }

        // Increment the image buffer pointer for the next row of tiles
        data += tileHeight*scanlineSize;
    }

    // Free the tile buffer
    _TIFFfree(tileBuf);

    return 1;
}

