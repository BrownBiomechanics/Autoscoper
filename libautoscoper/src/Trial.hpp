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

/// \file Trial.hpp
/// \author Andy Loomis

#ifndef XROMM_TRIAL_HPP
#define XROMM_TRIAL_HPP

#include <string>
#include <vector>

#include "Camera.hpp"
#include "CoordFrame.hpp"
#include "KeyCurve.hpp"
#include "Video.hpp"
#include "Volume.hpp"

namespace xromm
{
// The trial class contains all of the state information for an autoscoper run.
// It should eventually become an in-memory representation of the xromm
// autoscoper file format. Currently that file format does not however hold the
// tracking information.

class Trial
{
public:

    // Loads a trial file

    Trial(const std::string& filename = "");

    void save(const std::string& filename);

    std::vector<Camera> cameras;
    std::vector<Video>  videos;
    std::vector<Volume> volumes;

    // State information
    int frame;
    int num_frames;

    KeyCurve x_curve;
    KeyCurve y_curve;
    KeyCurve z_curve;
    KeyCurve yaw_curve;
    KeyCurve pitch_curve;
    KeyCurve roll_curve;

    //Controls for the optimization process
    int guess;
    double offsets[6];
    int render_width;
    int render_height;

    CoordFrame volumeTrans;
};

} // namespace xromm

#endif // XROMM_TRIAL_HPP
