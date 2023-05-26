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
#include <windows.h>
#include "Win32/dirent.h"
#elif __APPLE__
#include <sys/types.h>
#include <sys/dir.h>
#endif
#ifdef __linux__
#include <sys/types.h>
#include <sys/dir.h>
#endif

#include <stdexcept>

#include "Video.hpp"
#include "TiffImage.h"

namespace xromm
{

  Video::Video(const std::string& dirname)
    : dirname_(dirname),
    filenames_(),
    frame_(),
    image_(new TiffImage()),
    background_(NULL)
{
    DIR *dir = opendir(dirname_.c_str());
    if (dir == 0) {
        throw std::runtime_error("Unable to open directory: " + dirname_);
    }

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {

        // Ignore hidden files
        std::string filename(ent->d_name);
        if(filename.compare(0,1,".") == 0) {
            continue;
        }

        // Only search for tifs
        size_t ext = filename.find_last_of(".")+1;
        if (filename.compare(ext, 3, "tif") != 0) {
            continue;
        }

        filenames_.push_back(dirname + "/" + filename);
    }
    closedir(dir);

    // Check if we found any files
    if (filenames_.size() == 0) {
        throw std::runtime_error("No tif files found in directory: " + dirname_);
    }

    // Sort the filenames and load the first frame
    sort(filenames_.begin(), filenames_.end());
    set_frame(0);
}

Video::Video(const Video& video)
    : dirname_(video.dirname_),
      filenames_(video.filenames_),
      frame_(),
    background_(NULL)
{
    image_ = tiffImageCopy(video.image_);
  if (video.background_){
    background_ = new float[image_->width*image_->height];
    memcpy(background_, video.background_, image_->width * image_->height * sizeof(float));
  }
}

Video::~Video()
{
    if (image_) tiffImageFree(image_);
  if (background_) delete[] background_;
}

Video&
Video::operator=(const Video& video)
{
    dirname_   = video.dirname_;
    filenames_ = video.filenames_;
    frame_     = video.frame_;

    if (image_) tiffImageFree(image_);
    image_ = tiffImageCopy(video.image_);

  if (video.background_){
    if (background_) delete[] background_;
    background_ = new float[image_->width*image_->height];
    memcpy(background_, video.background_, image_->width * image_->height * sizeof(float));
  }
    return *this;
}

int Video::create_background_image()
{
  if (filenames_.size() < 2)
    return -1;


  if (background_) delete[] background_;
  background_ = new float[width()*height()];
  memset(background_, 0, width()*height()*sizeof(float));

  //Read tmp_image
  TiffImage* tmp_image = new TiffImage();
  TIFFSetWarningHandler(0);
  TIFF* tif;

  for (int i = 0; i < filenames_.size(); i++){
    tif = TIFFOpen(filenames_.at(i).c_str(), "r");
    if (!tif) {
      std::cerr << "Video::frame(): Unable to open image. " << std::endl;
      return -2;
    }

    tiffImageFree(tmp_image);
    tiffImageRead(tif, tmp_image);
    TIFFClose(tif);

    if (tmp_image->bitsPerSample == 8){
      unsigned char * ptr = reinterpret_cast<unsigned char*> (tmp_image->data);
      float* ptr_b = background_;
      for (; ptr < reinterpret_cast<unsigned char*>(tmp_image->data) + tmp_image->dataSize; ptr++, ptr_b++)
      {
        float val = *ptr;
        if (val / 255 > *ptr_b)
          *ptr_b = val / 255;
      }
    }
    else
    {
      unsigned short * ptr = static_cast<unsigned short*> (tmp_image->data);
      float * ptr_b = background_;
      for (; ptr < reinterpret_cast<unsigned short*>(tmp_image->data) + tmp_image->dataSize; ptr++, ptr_b++)
      {
        float val = *ptr;
        if (val / 65535  > *ptr_b)
          *ptr_b = val / 65535;
      }
    }
  }

  tiffImageFree(tmp_image);

  return 1;
}



  void
Video::set_frame(size_type i)
{
    if (i >= filenames_.size()) {
        i = filenames_.size()-1;
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

    frame_  = i;
}

Video::size_type
Video::width() const
{
    return image_->width;
}

Video::size_type
Video::height() const
{
    return image_->height;
}

Video::size_type
Video::bps() const
{
    return image_->bitsPerSample;
}

const void*
Video::data() const
{
    return image_->data;
}


} // namespace xromm

