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

#ifndef XROMM_VOLUME_TRANSFORM_HPP
#define XROMM_VOLUME_TRANSFORM_HPP

#include <vector>

#include "KeyCurve.hpp"
#include "CoordFrame.hpp"

namespace xromm {

// The trial class contains all of the state information for an autoscoper run.
// It should eventually become an in-memory representation of the xromm
// autoscoper file format. Currently that file format does not however hold the
// tracking information.

class VolumeTransform
{
public:
  // Loads a trial file
  VolumeTransform() { addCurveSet(); }
  ~VolumeTransform();

  KeyCurve<float>* getXCurve() { return x_curves[currentCurveSet]; }
  KeyCurve<float>* getYCurve() { return y_curves[currentCurveSet]; }
  KeyCurve<float>* getZCurve() { return z_curves[currentCurveSet]; }
  KeyCurve<Quatf>* getQuatCurve() { return quat_curves[currentCurveSet]; }

  size_t numberOfCurveSets() const { return x_curves.size(); }
  void setCurrentCurveSet(const int& idx);
  void addCurveSet();
  void setCurrentCurveSetToNext();
  void setCurrentCurveSetToPrevious();

  // CoordFrame volumeTrans; //FromWorldToVolume
  CoordFrame volumeMatrix; // FromWorldToPivot

private:
  int currentCurveSet;
  std::vector<KeyCurve<float>*> x_curves;
  std::vector<KeyCurve<float>*> y_curves;
  std::vector<KeyCurve<float>*> z_curves;
  std::vector<KeyCurve<Quatf>*> quat_curves;
};
} // namespace xromm

#endif // XROMM_VOLUME_TRANSFORM
