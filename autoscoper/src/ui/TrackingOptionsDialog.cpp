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

TrackingOptionsDialog::TrackingOptionsDialog(QWidget *parent) :
												QDialog(parent),
												diag(new Ui::TrackingOptionsDialog){

	diag->setupUi(this);
	doExit = false;
	frame_optimizing = false;
	from_frame = 0;
    to_frame = 0;
	curFrame = 0;
	d_frame = 1;
	num_repeats = 1;

	// Cost function index: Default is NCC
	opt_method = 0;

	// Backup Save
	is_backup_on = 1; // Always on

	// Neldon Optimization Parameters
	nm_opt_alpha = 1;
	nm_opt_gamma = 2;
	nm_opt_beta  = 0.5;

	// Read random search limits and iterations
	inner_iter = 40;
	trans_limit = 2;
	rot_limit = 4;


	inActive = false;
}

TrackingOptionsDialog::~TrackingOptionsDialog(){
	delete diag;
}


void TrackingOptionsDialog::frame_optimize()
{
	frame_optimizing = true;
	AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());

    if (!mainwindow || frame == -1) {

        return;
    }

    while (frame != to_frame+d_frame && !doExit) {

		if (mainwindow->getPosition_graph()->frame_locks.at(frame)) {
            frame += d_frame;
            return;
         }


		// Read Neldon Optimization Parameters
		nm_opt_alpha = diag->spinBox_alpha->value();
		nm_opt_gamma = diag->spinBox_gamma->value();
		nm_opt_beta  = diag->spinBox_beta->value() / 10;

		// Read random search limits and iterations
		inner_iter = diag->spinBox_inner_iter->value();
		rot_limit = diag->spinBox_rot_limit->value();
		trans_limit = diag->spinBox_trans_limit->value();

		// Optimization
		  mainwindow->getTracker()->optimize(frame, d_frame, num_repeats, nm_opt_alpha, nm_opt_gamma, nm_opt_beta, opt_method, inner_iter, rot_limit, trans_limit);

          mainwindow->update_graph_min_max(mainwindow->getPosition_graph(), mainwindow->getTracker()->trial()->frame);

		  mainwindow->setFrame(mainwindow->getTracker()->trial()->frame);
		 
         //update progress bar
		  
         double value = abs(to_frame-from_frame) > 0 ? abs(frame-from_frame)/ abs(to_frame-from_frame) : 0;
		 int progress = value *100;
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

void TrackingOptionsDialog::retrack(){
	if (from_frame != to_frame) {
        this->show();
    }

	if (!frame_optimizing) {
        frame = from_frame;
        doExit = false;
		frame_optimize();
    }
}

void TrackingOptionsDialog::trackCurrent() {

	AutoscoperMainWindow *mainwindow = dynamic_cast <AutoscoperMainWindow *> (parent());

	mainwindow->getTracker()->trial()->guess = 0;

	curFrame = mainwindow->getCurrentFrame();// Read Current Frame

	if (!mainwindow) return;

	if (!frame_optimizing) {
		doExit = false;
		from_frame = curFrame;
		to_frame = curFrame;
		frame = from_frame;
		frame_optimize();
	}

}

void TrackingOptionsDialog::setRange(int from, int to, int max){
	diag->spinBox_FrameStart->setMinimum(0);
	diag->spinBox_FrameStart->setMaximum(max);
	diag->spinBox_FrameStart->setValue(from);

	diag->spinBox_FrameEnd->setMinimum(0);
	diag->spinBox_FrameEnd->setMaximum(max);
	diag->spinBox_FrameEnd->setValue(to);
}

void TrackingOptionsDialog::on_pushButton_OK_clicked(bool checked){
	if(!inActive){
		AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());
		if(!mainwindow) return;

		from_frame = diag->spinBox_FrameStart->value();
		to_frame = diag->spinBox_FrameEnd->value();

 		num_repeats = diag->spinBox_NumberRefinements->value();
		int downhill_method = diag->radioButton_downhill_method->isChecked();
		if (downhill_method)
		{
			opt_method = 1; // runs downhill simplex
		}
		else {
			opt_method = 0; // runs random search method
		}

		bool reverse = diag->checkBox_Reverse->checkState() != Qt::Unchecked;

		mainwindow->push_state();

		if (mainwindow->getTracker()->trial()->guess == 0) {
			double xyzypr[6];
			(CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()))*(*mainwindow->getTracker()->trial()->getVolumeMatrix(-1))).to_xyzypr(xyzypr);

			mainwindow->getTracker()->trial()->getXCurve(-1)->insert(from_frame, xyzypr[0]);
			mainwindow->getTracker()->trial()->getYCurve(-1)->insert(from_frame, xyzypr[1]);
			mainwindow->getTracker()->trial()->getZCurve(-1)->insert(from_frame, xyzypr[2]);
			mainwindow->getTracker()->trial()->getYawCurve(-1)->insert(from_frame, xyzypr[3]);
			mainwindow->getTracker()->trial()->getPitchCurve(-1)->insert(from_frame, xyzypr[4]);
			mainwindow->getTracker()->trial()->getRollCurve(-1)->insert(from_frame, xyzypr[5]);
		}

		if (!frame_optimizing) {
			if (reverse) {
				frame = to_frame;
				int tmp = from_frame;
				from_frame = to_frame;
				to_frame = tmp;
				d_frame = from_frame > to_frame? -1: 1;
			}
			else {
				frame = from_frame;
				d_frame = from_frame > to_frame? -1: 1;
			}
			doExit = false;
			frame_optimize();
		}
	}else{
		this->accept();
	}
}

void TrackingOptionsDialog::on_pushButton_Cancel_clicked(bool checked){
	if(!inActive){
		doExit = true;
		if(!frame_optimizing)frame_optimize();
		frame_optimizing = false;
	}else{
		this->reject();
	}
}

void TrackingOptionsDialog::on_radioButton_CurrentFrame_clicked(bool checked){
	AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());

	mainwindow->getTracker()->trial()->guess = 0;
}
void TrackingOptionsDialog::on_radioButton_PreviousFrame_clicked(bool checked){
	AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());

	mainwindow->getTracker()->trial()->guess = 1;
}
void TrackingOptionsDialog::on_radioButton_LinearExtrapolation_clicked(bool checked){
	AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());

	mainwindow->getTracker()->trial()->guess = 2;
}
void TrackingOptionsDialog::on_radioButton_SplineInterpolation_clicked(bool checked){
	AutoscoperMainWindow *mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent());

	mainwindow->getTracker()->trial()->guess = 3;
}