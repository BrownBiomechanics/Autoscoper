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

/// \file GLView.h
/// \author Benjamin Knorlein, Andy Loomis


#ifndef GL_VIEW_H_
#define GL_VIEW_H_

#include <QGLWidget>
#include "ui/GLWidget.h"
#include "CoordFrame.hpp"

namespace xromm{
  namespace gpu{
    class View;
  }
}
using xromm::gpu::View;
using xromm::CoordFrame;

class QMouseEvent;
class QWheelEvent;

class GLView: public GLWidget
{
    Q_OBJECT

public:
    GLView(QWidget *parent);

  void setView(View * view);
  void setStaticView(bool staticView);
  void saveView(std::string filename);

protected:
    void paintGL();
  void mousePressEvent(QMouseEvent *e);
  void mouseMoveEvent(QMouseEvent *e);
  void mouseReleaseEvent(QMouseEvent *e);
  void wheelEvent(QWheelEvent *e);
private:
  View * m_view;
  CoordFrame * volume_matrix;

  // Default camera
  CoordFrame defaultViewMatrix;

  double press_x;
  double press_y;
  double prevx;
    double prevy;

  void select_manip_in_view(double x, double y, int button);
  void draw_manip_from_view(const ViewData* view);

  void move_manip_in_view(double x, double y, bool out_of_plane=false);
  void update_scale_in_view(ViewData* view);
  void enable_headlight();

  void draw_gradient(const float* top_color, const float* bot_color);
  void draw_xz_grid(int width, int height, float scale);
  void draw_cylinder(float radius, float height, int slices);
  void draw_camera();
  void draw_textured_quad(const double* pts, unsigned int texid);
};



#endif /* GL_VIEW_H_ */
