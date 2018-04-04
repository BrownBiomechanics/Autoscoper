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

/// \file Video.hpp
/// \author Andy Loomis

#ifndef XROMM_VIDEO_HPP
#define XROMM_VIDEO_HPP

#include <string>
#include <vector>

struct TiffImage;

namespace xromm
{

// Encapsulates an x ray video sequence. It is created from a directory of
// image frames ordered alphanumerically. It loads each frame on demand, and
// does not store the entire video in memory.

class Video
{
public:

    typedef std::vector<std::string>::size_type size_type;

    // Loads a directory of video frames -- only supports TIFF

    Video(const std::string& dirname);

    Video(const Video& video);

    ~Video();

    Video& operator=(const Video& video);

    // Accessors

	int create_background_image();

    const std::string& dirname() const { return dirname_; }

    const std::string& filename(size_type i) const { return filenames_.at(i); }

    size_type num_frames() const { return filenames_.size(); }

    void set_frame(size_type i);

    size_t frame() const { return frame_; }

    // Frame information

    size_type width() const;

    size_type height() const;

    size_type bps() const;

    const void* data() const;

	const float* background() const { return background_; }

private:

    std::string dirname_;

    std::vector<std::string> filenames_;

    size_type frame_;

    TiffImage* image_;

	float* background_;
};

} // namespace xromm

#endif // XROMM_VIDEO_HPP
