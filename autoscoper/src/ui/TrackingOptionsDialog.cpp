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
	d_frame = 1;
	num_repeats = 1;
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

		  mainwindow->getTracker()->optimize(frame, d_frame, num_repeats);

          mainwindow->update_graph_min_max(mainwindow->getPosition_graph(), mainwindow->getTracker()->trial()->frame);

		  mainwindow->setFrame(mainwindow->getTracker()->trial()->frame);
		 
         //update progress bar
		  
         double value = abs(to_frame-from_frame) > 0 ? abs(frame-from_frame)/ abs(to_frame-from_frame) : 0;
		 int progress = value *100;
		 diag->progressBar->setValue(progress);

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

		bool reverse = diag->checkBox_Reverse->checkState() != Qt::Unchecked;

		mainwindow->getTracker()->trial()->volumeTrans = *mainwindow->getVolume_matrix();

		mainwindow->push_state();

		if (mainwindow->getTracker()->trial()->guess == 0) {
			double xyzypr[6];
			(CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()))*(*mainwindow->getVolume_matrix())).to_xyzypr(xyzypr);

			mainwindow->getTracker()->trial()->x_curve.insert(from_frame,xyzypr[0]);
			mainwindow->getTracker()->trial()->y_curve.insert(from_frame,xyzypr[1]);
			mainwindow->getTracker()->trial()->z_curve.insert(from_frame,xyzypr[2]);
			mainwindow->getTracker()->trial()->yaw_curve.insert(from_frame,xyzypr[3]);
			mainwindow->getTracker()->trial()->pitch_curve.insert(from_frame,xyzypr[4]);
			mainwindow->getTracker()->trial()->roll_curve.insert(from_frame,xyzypr[5]);
		}

		if (!frame_optimizing) {
			if (reverse) {
				frame = to_frame;
				int tmp = from_frame;
				from_frame = to_frame;
				to_frame = tmp;
				d_frame = from_frame > to_frame? 1: -1;
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