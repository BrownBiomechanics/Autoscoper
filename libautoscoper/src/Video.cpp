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

#include <stdexcept>

#include "Video.hpp"
#include "TiffImage.h"

using namespace std;

namespace xromm
{

	Video::Video(const string& dirname)
		: dirname_(dirname),
		filenames_(),
		frame_(),
		image_(new TiffImage()),
		background_(new TiffImage())
{
    DIR *dir = opendir(dirname_.c_str());
    if (dir == 0) {
        throw runtime_error("Unable to open directory: " + dirname_);
    }

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {

        // Ignore hidden files
        string filename(ent->d_name);
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
        throw runtime_error("No tif files found in directory: " + dirname_);
    }

    // Sort the filenames and load the first frame
    sort(filenames_.begin(), filenames_.end());
    set_frame(0);
}

Video::Video(const Video& video)
    : dirname_(video.dirname_),
      filenames_(video.filenames_),
      frame_()
{
    image_ = tiffImageCopy(video.image_);
	background_ = tiffImageCopy(video.background_);
}

Video::~Video()
{
    if (image_) tiffImageFree(image_);
	if (background_)tiffImageFree(background_);
}

Video&
Video::operator=(const Video& video)
{
    dirname_   = video.dirname_;
    filenames_ = video.filenames_;
    frame_     = video.frame_;

    if (image_) tiffImageFree(image_);
    image_ = tiffImageCopy(video.image_);

	if (background_)tiffImageFree(background_);
	background_ = tiffImageCopy(video.background_);

    return *this;
}

int Video::create_background_image()
{
	if (filenames_.size() < 2)
		return -1;

	TiffImage* tmp_image = new TiffImage();
	//Read tmp_image
	TIFFSetWarningHandler(0);

	//Read _background
	TIFF* tif = TIFFOpen(filenames_.at(0).c_str(), "r");
	
	if (!tif) {
		cerr << "Video::frame(): Unable to open image. " << endl;
		return -2;
	}

	tiffImageFree(background_);
	tiffImageRead(tif, background_);
	TIFFClose(tif);

	memset(background_->data, 0, background_->dataSize);
	tif = TIFFOpen("background.tif", "w");
	tiffImageWrite(tif, background_);
	TIFFClose(tif);
	for (int i = 0; i < filenames_.size(); i++){
		tif = TIFFOpen(filenames_.at(i).c_str(), "r");
		if (!tif) {
			cerr << "Video::frame(): Unable to open image. " << endl;
			return -2;
		}

		tiffImageFree(tmp_image);
		tiffImageRead(tif, tmp_image);
		TIFFClose(tif);

		if (background_->bitsPerSample == 8){
			unsigned char * ptr = reinterpret_cast<unsigned char*> (tmp_image->data);
			unsigned char * ptr_b = reinterpret_cast<unsigned char*> (background_->data);
			for (; ptr < reinterpret_cast<unsigned char*>(tmp_image->data) + tmp_image->dataSize; ptr++, ptr_b++)
			{
				if (*ptr > *ptr_b)
					*ptr_b = *ptr;
			}
		}
		else
		{
			unsigned short * ptr = static_cast<unsigned short*> (tmp_image->data);
			unsigned short * ptr_b = reinterpret_cast<unsigned short*> (background_->data);
			for (; ptr < reinterpret_cast<unsigned short*>(tmp_image->data) + tmp_image->dataSize; ptr++, ptr_b++)
			{
				if (*ptr != 0 && *ptr_b == 0)
					*ptr_b = 255;
			}
		}

		tif = TIFFOpen("background.tif", "w");
		tiffImageWrite(tif, background_);
		TIFFClose(tif);
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
        cerr << "Video::frame(): Unable to open image. " << endl;
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

