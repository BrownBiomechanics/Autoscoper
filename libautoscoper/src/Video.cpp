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
// ----------------------------------

/// \file Video.cpp
/// \author Andy Loomis, Benjamin Knorlein

#include <cstring>
#include <algorithm>
#include <iostream>
#ifdef WIN32
#  ifndef NOMINMAX // Otherwise windows.h defines min/max macros
#    define NOMINMAX
#  endif // NOMINMAX
#  include <windows.h>
#  include "Win32/dirent.h"
#elif __APPLE__
#  include <sys/types.h>
#  include <sys/dir.h>
#endif
#ifdef __linux__
#  include <sys/types.h>
#  include <sys/dir.h>
#endif

#include <stdexcept>

#include "Video.hpp"
#include "TiffImage.h"

namespace xromm {
Video::Video(const std::string& dirname)
  : dirname_(dirname)
  , filenames_()
  , frame_()
  , image_(new TiffImage())
  , background_(NULL)
{
  DIR* dir = opendir(dirname_.c_str());
  if (dir == 0) {
    throw std::runtime_error("Unable to open directory: " + dirname_);
  }

  struct dirent* ent;
  while ((ent = readdir(dir)) != NULL) {

    // Ignore hidden files
    std::string filename(ent->d_name);
    if (filename.compare(0, 1, ".") == 0) {
      continue;
    }

    // Only search for tifs
    size_t ext = filename.find_last_of(".") + 1;
    if (filename.compare(ext, 3, "tif") != 0) {
      continue;
    }

    filenames_.push_back(dirname + "/" + filename);
  }
  closedir(dir);

  // Check if we found any files
  if (filenames_.size() == 0) {
    throw std::runtime_error("No TIFF files found in directory: " + dirname_);
  }

  // Sort the filenames and load the first frame
  sort(filenames_.begin(), filenames_.end());
  set_frame(0);
}

Video::Video(const Video& video)
  : dirname_(video.dirname_)
  , filenames_(video.filenames_)
  , frame_()
  , background_(NULL)
{
  image_ = tiffImageCopy(video.image_);
  if (video.background_) {
    background_ = new float[image_->width * image_->height];
    memcpy(background_, video.background_, image_->width * image_->height * sizeof(float));
  }
}

Video::~Video()
{
  if (image_)
    tiffImageFree(image_);
  if (background_)
    delete[] background_;
}

Video& Video::operator=(const Video& video)
{
  dirname_ = video.dirname_;
  filenames_ = video.filenames_;
  frame_ = video.frame_;

  if (image_)
    tiffImageFree(image_);
  image_ = tiffImageCopy(video.image_);

  if (video.background_) {
    if (background_)
      delete[] background_;
    background_ = new float[image_->width * image_->height];
    memcpy(background_, video.background_, image_->width * image_->height * sizeof(float));
  }
  return *this;
}

bool Video::create_background_image()
{
  if (filenames_.size() < 2) {
    std::cerr << "Video::create_background_image(): Not enough images to create background image." << std::endl;
    return false;
  }

  if (background_)
    delete[] background_;
  background_ = new float[width() * height()];
  memset(background_, 0, width() * height() * sizeof(float));

  for (size_t i = 0; i < filenames_.size(); i++) {

    // Read tmp_image
    TIFFSetWarningHandler(0);
    TIFF* tif = TIFFOpen(filenames_.at(i).c_str(), "r");

    if (!tif) {
      std::cerr << "Video::create_background_image(): Unable to open image: " << filenames_.at(i) << std::endl;
      return false;
    }

    TiffImage* tmp_image = new TiffImage();
    tiffImageRead(tif, tmp_image);
    TIFFClose(tif);

    switch (tmp_image->bitsPerSample) {
      case 8:
        create_background_image_internal<unsigned char>(tmp_image);
        tiffImageFree(tmp_image);
        break;
      case 16:
        create_background_image_internal<unsigned short>(tmp_image);
        tiffImageFree(tmp_image);
        break;
      case 32:
        create_background_image_internal<unsigned int>(tmp_image);
        tiffImageFree(tmp_image);
        break;
      default:
        std::cerr << "Video::create_background_image(): Unsupported bits per sample." << std::endl;
        tiffImageFree(tmp_image);
        return false;
    }
  }

  return true;
}

template <typename T>
void Video::create_background_image_internal(TiffImage* tmp_img)
{
  static_assert(std::is_same<T, unsigned char>::value || std::is_same<T, unsigned short>::value
                  || std::is_same<T, unsigned int>::value,
                "T must be of type unsigned char, unsigned short, or unsigned int");
  constexpr unsigned int normalization_factor = std::numeric_limits<T>::max();
  static_assert(normalization_factor == 255 || normalization_factor == 65535 || normalization_factor == 4294967295,
                "normalization_factor must be one of 255, 65535 or 4294967295");
  T* start = reinterpret_cast<T*>(tmp_img->data);
  T* end = start + (tmp_img->dataSize / sizeof(T));
  T* iter = start;
  float* bg = background_;
  while (iter < end) {
    float val = *iter;
    if (val / normalization_factor > *bg) {
      *bg = val / normalization_factor;
    }
    iter++;
    bg++;
  }
}

void Video::set_frame(size_type i)
{
  if (i >= filenames_.size()) {
    i = filenames_.size() - 1;
  }

  TIFFSetWarningHandler(0);
  TIFF* tif = TIFFOpen(filenames_.at(i).c_str(), "r");
  if (!tif) {
    std::cerr << "Video::frame(): Unable to open image. " << std::endl;
    return;
  }

  tiffImageFree(image_);
  tiffImageRead(tif, image_);

  TIFFClose(tif);

  frame_ = i;
}

Video::size_type Video::width() const
{
  return image_->width;
}

Video::size_type Video::height() const
{
  return image_->height;
}

Video::size_type Video::bps() const
{
  return image_->bitsPerSample;
}

const void* Video::data() const
{
  return image_->data;
}
} // namespace xromm
