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

/// \file AdvancedOptionsDialog.cpp
/// \author Benjamin Knorlein, Andy Loomis

#include "ui/AdvancedOptionsDialog.h"
#include "ui_AdvancedOptionsDialog.h"
#include "ui/AutoscoperMainWindow.h"
#include "ui/TimelineDockWidget.h"

#include "Tracker.hpp"
#include "Trial.hpp"
#include "Manip3D.hpp"

AdvancedOptionsDialog::AdvancedOptionsDialog(QWidget *parent) :
                        QDialog(parent),
                        adv_diag(new Ui::AdvancedOptionsDialog){

  adv_diag->setupUi(this);
  doExit = false;

  from_frame = 0;
    to_frame = 0;
  skip_frame = 1;

  curFrame = 0;
  d_frame = 1;

  // Smoothing
  winSizeSmoothing = 5;

  inActive = false;
}

AdvancedOptionsDialog::~AdvancedOptionsDialog(){
  delete adv_diag;
}

void AdvancedOptionsDialog::setRangeAdvanced(int from, int to, int max){
  adv_diag->spinBox_FrameStart_adv->setMinimum(0);
  adv_diag->spinBox_FrameStart_adv->setMaximum(max);
  adv_diag->spinBox_FrameStart_adv->setValue(from);

  adv_diag->spinBox_FrameEnd_adv->setMinimum(0);
  adv_diag->spinBox_FrameEnd_adv->setMaximum(max);
  adv_diag->spinBox_FrameEnd_adv->setValue(to);
}

void AdvancedOptionsDialog::on_pushButton_Delete_clicked(bool checked) {
  if (!inActive) {
    AutoscoperMainWindow *mainwindow = dynamic_cast <AutoscoperMainWindow *> (parent());
    if (!mainwindow) return;

    from_frame = adv_diag->spinBox_FrameStart_adv->value();
    to_frame = adv_diag->spinBox_FrameEnd_adv->value();
    skip_frame = adv_diag->spinBox_FrameSkip_adv->value();

    for (int iF = from_frame; iF <= to_frame; iF++)
    {
      mainwindow->deletePose(iF);
    }

    puts("Poses were deleted.");

    mainwindow->push_state();

    mainwindow->update_xyzypr_and_coord_frame();
    mainwindow->redrawGL();
  }
  else {
    this->accept();
  }
}

void AdvancedOptionsDialog::on_pushButton_Smooth_clicked(bool checked) {
  if (!inActive) {
    AutoscoperMainWindow *mainwindow = dynamic_cast <AutoscoperMainWindow *> (parent());
    if (!mainwindow) return;

    from_frame = adv_diag->spinBox_FrameStart_adv->value();
    to_frame = adv_diag->spinBox_FrameEnd_adv->value();
    skip_frame = adv_diag->spinBox_FrameSkip_adv->value();
    winSizeSmoothing = adv_diag->spinBox_winSize->value();

    mainwindow->MovingAverageFilter(winSizeSmoothing, from_frame, to_frame);

    puts("Poses were smoothed using moving average filter.");

    mainwindow->push_state();

    mainwindow->update_xyzypr_and_coord_frame();
    mainwindow->redrawGL();
  }
  else {
    this->accept();
  }
}

void AdvancedOptionsDialog::on_radioButton_MovingAverage_clicked(bool checked){
  // AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());

  //mainwindow->getTracker()->trial()->guess = 0;
}
void AdvancedOptionsDialog::on_radioButton_AnotherMethod_clicked(bool checked){
  // AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());

  //mainwindow->getTracker()->trial()->guess = 1;
}

void AdvancedOptionsDialog::setDefPaths(QString root_path, QString filter_folder, QString filter_name, QString tracking_folder, QString task_name) {

  adv_diag->lineEdit_rootPath->setText(root_path);
  adv_diag->lineEdit_filterFolder->setText(filter_folder);
  adv_diag->lineEdit_filterName->setText(filter_name);
  adv_diag->lineEdit_trackingPath->setText(tracking_folder);
  adv_diag->lineEdit_TaskName->setText(task_name);
}
