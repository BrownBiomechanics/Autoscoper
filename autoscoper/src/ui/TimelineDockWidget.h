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

/// \file TimelineDockWidget.h
/// \author Benjamin Knorlein, Andy Loomis

#ifndef TIMELINEDOCKWIDGET_H
#define TIMELINEDOCKWIDGET_H

#include <QDockWidget>
#include "KeyCurve.hpp"

struct GraphData
{
  bool show_x;
  bool show_y;
  bool show_z;
  bool show_yaw;
  bool show_pitch;
  bool show_roll;

  double min_frame;
  double max_frame;
  double min_value;
  double max_value;

  std::vector<bool> frame_locks;
};

enum Selection_type { NODE, IN_TANGENT, OUT_TANGENT };

// forward declarations
namespace Ui {
class TimelineDockWidget;
}

class QOpenGLContext;
class AutoscoperMainWindow;

class TimelineDockWidget : public QDockWidget
{

  Q_OBJECT

public:
  explicit TimelineDockWidget(QWidget* parent = 0);
  ~TimelineDockWidget();

  void setFramesRange(int firstFrame, int lastFrame );
  GraphData* getPosition_graph() { return position_graph; }
  AutoscoperMainWindow* getMainWindow() { return mainwindow; };
  std::vector<std::pair<std::pair<KeyCurve*, KeyCurve::iterator>, Selection_type>>* getSelectedNodes() { return &selected_nodes; }
  void setSelectedNodes(std::vector<std::pair<std::pair<KeyCurve*, KeyCurve::iterator>, Selection_type>> new_nodes) { selected_nodes = new_nodes; }

  std::vector<std::pair<KeyCurve*, KeyCurve::iterator>>* getCopiedNodes() { return &copied_nodes; }

  void getValues(double* xyzypr);
  void setValues(double* xyzypr);
  void setValuesEnabled(bool enabled);
  void setFrame(int frame);

  void update_graph_min_max(int frame = -1);

  void setTrial(Trial* trial);
  void draw();
  void setSpinButtonUpdate(bool spinButtonUpdate) { m_spinButtonUpdate = spinButtonUpdate; }

private:
  Ui::TimelineDockWidget* dock;
  GraphData* position_graph;

  AutoscoperMainWindow* mainwindow;
  bool m_spinButtonUpdate;

  std::vector<std::pair<std::pair<KeyCurve*, KeyCurve::iterator>, Selection_type>> selected_nodes;
  std::vector<std::pair<KeyCurve*, KeyCurve::iterator>> copied_nodes;

  // Video play
  int play_tag;
  QTimer* play_timer;

protected:

public slots:
  void play_update();

  void on_toolButton_PreviousFrame_clicked();
  void on_toolButton_Stop_clicked();
  void on_toolButton_Play_clicked();
  void on_toolButton_NextFrame_clicked();
  void on_toolButton_NextTenFrame_clicked();
  void on_horizontalSlider_Frame_valueChanged(int value);

  void on_doubleSpinBox_X_valueChanged ( double d );
  void on_doubleSpinBox_Y_valueChanged ( double d );
  void on_doubleSpinBox_Z_valueChanged ( double d );
  void on_doubleSpinBox_Yaw_valueChanged ( double d );
  void on_doubleSpinBox_Pitch_valueChanged ( double d );
  void on_doubleSpinBox_Roll_valueChanged ( double d );

  void on_checkBox_X_stateChanged ( int state );
  void on_checkBox_Y_stateChanged ( int state );
  void on_checkBox_Z_stateChanged ( int state );
  void on_checkBox_Yaw_stateChanged ( int state );
  void on_checkBox_Pitch_stateChanged ( int state );
  void on_checkBox_Roll_stateChanged ( int state );

  void on_spinBox_FirstFrame_valueChanged ( int d );
  void on_spinBox_LastFrame_valueChanged ( int d );

};

#endif  // TIMELINEDOCKWIDGET_H
