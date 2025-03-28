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

/// \file TrackingOptionsDialog.cpp
/// \author Benjamin Knorlein, Andy Loomis

#include "ui/TrackingOptionsDialog.h"
#include "ui_TrackingOptionsDialog.h"
#include "ui/AutoscoperMainWindow.h"
#include "ui/TimelineDockWidget.h"

#include "Tracker.hpp"
#include "Trial.hpp"
#include "Manip3D.hpp"

TrackingOptionsDialog::TrackingOptionsDialog(QWidget* parent)
  : QDialog(parent)
  , diag(new Ui::TrackingOptionsDialog)
{

  diag->setupUi(this);
  doExit = false;
  frame_optimizing = false;
  from_frame = 0;
  to_frame = 0;
  skip_frame = 1;
  curFrame = 0;
  d_frame = 1;
  num_repeats = 1;

  // Opt Method: Default is PSO
  opt_method = 0;

  // Backup Save
  is_backup_on = 1; // Always on

  // Read random search limits and iterations
  max_iter = 1000;
  min_lim = -3.0;
  max_lim = 3.0;
  max_stall_iter = 25;

  // Cost Function: Default is Bone Models
  cf_model = 0; // 0 is Bone Model ------ 1 is Implant Model

  inActive = false;
}

TrackingOptionsDialog::~TrackingOptionsDialog()
{
  delete diag;
}

void TrackingOptionsDialog::frame_optimize()
{
  frame_optimizing = true;
  AutoscoperMainWindow* mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent());

  if (!mainwindow || frame == -1) {

    return;
  }

  while (frame != to_frame + d_frame && !doExit) {

    // Check if frame is out of bound
    if (to_frame > from_frame) {
      if (frame > to_frame) {
        frame = to_frame;
      } else if (frame < from_frame) {
        frame = from_frame;
      }
    } else if (to_frame <= from_frame) {
      if (frame < to_frame) {
        frame = to_frame;
      } else if (frame >= from_frame) {
        frame = from_frame;
      }
    }

    //

    if (mainwindow->getPosition_graph()->frame_locks.at(frame)) {
      frame += d_frame;
      return;
    }

    // Read random search limits and iterations
    max_iter = diag->spinBox_max_iter->value();
    min_lim = diag->spinBox_min_lim->value();
    max_lim = diag->spinBox_max_lim->value();
    max_stall_iter = diag->spinBox_stall_max->value();

    // Set optimization parameters (SEND THEM TO TRACKER CLASS)
    /*
       cf_model =


     */

    // Optimization
    mainwindow->getTracker()->optimize(
      frame, d_frame, num_repeats, opt_method, max_iter, min_lim, max_lim, cf_model, max_stall_iter);

    mainwindow->update_graph_min_max(mainwindow->getPosition_graph(), mainwindow->getTracker()->trial()->frame);

    mainwindow->setFrame(mainwindow->getTracker()->trial()->frame);

    // update progress bar
    double value = abs(to_frame - from_frame) > 0 ? fabs(frame - from_frame) / fabs(to_frame - from_frame) : 0;
    int progress = value * 100;
    diag->progressBar->setValue(progress);

    mainwindow->backup_tracking(is_backup_on); // save backup before finishing optimization

    frame += d_frame;

    QApplication::processEvents();
  }

  frame_optimizing = false;
  QApplication::processEvents();
  diag->progressBar->setValue(0);
  this->close();
}

/*void TrackingOptionsDialog::retrack(){
   if (from_frame != to_frame) {
        this->show();
    }

   if (!frame_optimizing) {
        frame = from_frame;
        doExit = false;
    frame_optimize();
    }
   }*/

void TrackingOptionsDialog::trackCurrent()
{

  AutoscoperMainWindow* mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent());

  mainwindow->getTracker()->trial()->guess = 0;

  curFrame = mainwindow->getCurrentFrame(); // Read Current Frame

  if (!mainwindow)
    return;

  if (!frame_optimizing) {
    doExit = false;
    from_frame = curFrame;
    to_frame = curFrame;
    frame = from_frame;
    frame_optimize();
  }
}

void TrackingOptionsDialog::setRange(int from, int to, int max)
{
  diag->spinBox_FrameStart->setMinimum(0);
  diag->spinBox_FrameStart->setMaximum(max);
  diag->spinBox_FrameStart->setValue(from);

  diag->spinBox_FrameEnd->setMinimum(0);
  diag->spinBox_FrameEnd->setMaximum(max);
  diag->spinBox_FrameEnd->setValue(to);
}

void TrackingOptionsDialog::on_pushButton_OK_clicked(bool checked)
{
  if (!inActive) {
    AutoscoperMainWindow* mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent());
    if (!mainwindow)
      return;

    from_frame = diag->spinBox_FrameStart->value();
    to_frame = diag->spinBox_FrameEnd->value();
    skip_frame = diag->spinBox_FrameSkip->value();

    num_repeats = diag->spinBox_NumberRefinements->value();

    // Read Opt Method
    int downhill_method = diag->radioButton_downhill_method->isChecked();
    if (downhill_method) {
      opt_method = 1; // runs downhill simplex
    } else {
      opt_method = 0; // runs particle swarm optimization
    }

    // Read Cost Function
    int cf_model_int = diag->radioButton_bone_ncc_cf->isChecked();
    if (cf_model_int) {
      cf_model = 0; // runs bone
    } else {
      cf_model = 1; // runs implant
    }

    bool reverse = diag->checkBox_Reverse->checkState() != Qt::Unchecked;

    mainwindow->push_state();

    if (mainwindow->getTracker()->trial()->guess == 0) {
      double xyzypr[6];
      (CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()))
       * (*mainwindow->getTracker()->trial()->getVolumeMatrix(-1)))
        .to_xyzypr(xyzypr);

      mainwindow->getTracker()->trial()->getXCurve(-1)->insert(from_frame, xyzypr[0]);
      mainwindow->getTracker()->trial()->getYCurve(-1)->insert(from_frame, xyzypr[1]);
      mainwindow->getTracker()->trial()->getZCurve(-1)->insert(from_frame, xyzypr[2]);
      mainwindow->getTracker()->trial()->getQuatCurve(-1)->insert(from_frame, Quatf(xyzypr[3], xyzypr[4], xyzypr[5]));
    }

    if (!frame_optimizing) {
      if (reverse) {
        frame = to_frame;
        int tmp = from_frame;
        from_frame = to_frame;
        to_frame = tmp;
        d_frame = from_frame > to_frame ? -skip_frame : skip_frame;
      } else {
        frame = from_frame;
        d_frame = from_frame > to_frame ? -skip_frame : skip_frame;
      }
      doExit = false;

      frame_optimize();
    }
  } else {
    this->accept();
  }
}

void TrackingOptionsDialog::on_pushButton_Cancel_clicked(bool checked)
{
  if (!inActive) {
    doExit = true;
    if (!frame_optimizing)
      frame_optimize();
    frame_optimizing = false;
  } else {
    this->reject();
  }
}

void TrackingOptionsDialog::on_radioButton_CurrentFrame_clicked(bool checked)
{
  AutoscoperMainWindow* mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent());

  mainwindow->getTracker()->trial()->guess = 0;
}
void TrackingOptionsDialog::on_radioButton_PreviousFrame_clicked(bool checked)
{
  AutoscoperMainWindow* mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent());

  mainwindow->getTracker()->trial()->guess = 1;
}
void TrackingOptionsDialog::on_radioButton_LinearExtrapolation_clicked(bool checked)
{
  AutoscoperMainWindow* mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent());

  mainwindow->getTracker()->trial()->guess = 2;
}
void TrackingOptionsDialog::on_radioButton_SplineInterpolation_clicked(bool checked)
{
  AutoscoperMainWindow* mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent());

  mainwindow->getTracker()->trial()->guess = 3;
}
