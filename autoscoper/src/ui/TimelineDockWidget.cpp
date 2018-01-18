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

/// \file TimelineDockWidget.cpp
/// \author Benjamin Knorlein, Andy Loomis

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui_TimelineDockWidget.h"
#include "ui/TimelineDockWidget.h"
#include "ui/AutoscoperMainWindow.h"

#include "Tracker.hpp"

#include <QGLContext>
#include <QTimer>
#include <math.h>

TimelineDockWidget::TimelineDockWidget(QWidget *parent) :
										QDockWidget(parent),
										dock(new Ui::TimelineDockWidget){
	dock->setupUi(this);
	
	mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent);

	position_graph = new GraphData();
	position_graph->show_x = true;
    position_graph->show_y = true;
    position_graph->show_z = true;
    position_graph->show_yaw = true;
    position_graph->show_pitch = true;
    position_graph->show_roll = true;
    position_graph->min_frame = 0.0;
    position_graph->max_frame = 100.0;
    position_graph->min_value = -180.0;
    position_graph->max_value = 180.0;
    position_graph->frame_locks.resize(100,false);

	dock->gltimeline->setGraphData(position_graph);

	m_spinButtonUpdate = true;

	play_tag = 0;
	play_timer = new QTimer(this);
    connect(play_timer, SIGNAL(timeout()), this, SLOT(play_update()));
}

TimelineDockWidget::~TimelineDockWidget(){
	delete dock;
	delete position_graph;
}

void TimelineDockWidget::setTrial(Trial* trial){
	dock->gltimeline->setTrial(trial);
}

void TimelineDockWidget::draw(){
	dock->gltimeline->update();
}

void TimelineDockWidget::setSharedGLContext(const QGLContext * sharedContext){
	QGLContext* context = new QGLContext(sharedContext->format(), dock->gltimeline);
	context->create(sharedContext);
	dock->gltimeline->setContext(context,sharedContext,true);
}

void TimelineDockWidget::setFramesRange(int firstFrame, int lastFrame ){
	dock->spinBox_FirstFrame->setMinimum(firstFrame);
	dock->spinBox_FirstFrame->setMaximum(lastFrame);
	dock->spinBox_FirstFrame->setValue(firstFrame);

	dock->spinBox_LastFrame->setMinimum(firstFrame);
	dock->spinBox_LastFrame->setMaximum(lastFrame);
	dock->spinBox_LastFrame->setValue(lastFrame);

	dock->horizontalSlider_Frame->setRange(firstFrame,lastFrame);
	dock->horizontalSlider_Frame->setValue(firstFrame);
	dock->horizontalSlider_Frame->setPageStep(1);
	dock->labelFrame->setText(QString::number(firstFrame));
}

void TimelineDockWidget::on_toolButton_PreviousFrame_clicked(){
	dock->horizontalSlider_Frame->setValue(dock->labelFrame->text().toInt() - 1);
}
void TimelineDockWidget::on_toolButton_Stop_clicked(){
	if (play_tag) {
        play_tag = 0;
		play_timer->stop();
    }
}

void TimelineDockWidget::play_update(){
	on_toolButton_NextFrame_clicked();
	QApplication::processEvents();
}

void TimelineDockWidget::on_toolButton_Play_clicked(){
	 if (!play_tag) {
        play_tag = 1;
		play_timer->start(100);
    }
}
void TimelineDockWidget::on_toolButton_NextFrame_clicked(){
	dock->horizontalSlider_Frame->setValue(dock->labelFrame->text().toInt() + 1);
	if(play_tag && dock->horizontalSlider_Frame->value() == dock->horizontalSlider_Frame->maximum())on_toolButton_Stop_clicked();
}
void TimelineDockWidget::setFrame(int frame){
	dock->horizontalSlider_Frame->setValue(frame);
}

void TimelineDockWidget::on_horizontalSlider_Frame_valueChanged(int value){
	dock->labelFrame->setText(QString::number(value));
	
	mainwindow->setFrame(dock->labelFrame->text().toInt());
}

void TimelineDockWidget::on_doubleSpinBox_X_valueChanged ( double d ){
	if(m_spinButtonUpdate){
		mainwindow->update_coord_frame();
        update_graph_min_max();

        mainwindow->redrawGL();
	}
}

void TimelineDockWidget::on_doubleSpinBox_Y_valueChanged ( double d ){
	if(m_spinButtonUpdate){
		mainwindow->update_coord_frame();
        update_graph_min_max();

        mainwindow->redrawGL();
	}
}

void TimelineDockWidget::on_doubleSpinBox_Z_valueChanged ( double d ){
	if(m_spinButtonUpdate){
		mainwindow->update_coord_frame();
        update_graph_min_max();

        mainwindow->redrawGL();
	}
}

void TimelineDockWidget::on_doubleSpinBox_Yaw_valueChanged ( double d ){
	if(m_spinButtonUpdate){
		mainwindow->update_coord_frame();
        update_graph_min_max();

        mainwindow->redrawGL();
	}
}

void TimelineDockWidget::on_doubleSpinBox_Pitch_valueChanged ( double d ){
	if(m_spinButtonUpdate){
		mainwindow->update_coord_frame();
        update_graph_min_max();

        mainwindow->redrawGL();
	}
}

void TimelineDockWidget::on_doubleSpinBox_Roll_valueChanged ( double d ){
	if(m_spinButtonUpdate){
		mainwindow->update_coord_frame();
        update_graph_min_max();

        mainwindow->redrawGL();
	}
}


void TimelineDockWidget::on_checkBox_X_stateChanged ( int state ){
	position_graph->show_x = (state != Qt::Unchecked);
    update_graph_min_max();
    mainwindow->redrawGL();
}

void TimelineDockWidget::on_checkBox_Y_stateChanged ( int state ){
	position_graph->show_y = (state != Qt::Unchecked);
    update_graph_min_max();
    mainwindow->redrawGL();
}

void TimelineDockWidget::on_checkBox_Z_stateChanged ( int state ){
	position_graph->show_z = (state != Qt::Unchecked);
    update_graph_min_max();
    mainwindow->redrawGL();
}

void TimelineDockWidget::on_checkBox_Yaw_stateChanged ( int state ){
	position_graph->show_yaw = (state != Qt::Unchecked);
    update_graph_min_max();
    mainwindow->redrawGL();
}

void TimelineDockWidget::on_checkBox_Pitch_stateChanged ( int state ){
	position_graph->show_pitch = (state != Qt::Unchecked);
    update_graph_min_max();
    mainwindow->redrawGL();
}

void TimelineDockWidget::on_checkBox_Roll_stateChanged ( int state ){
	position_graph->show_roll = (state != Qt::Unchecked);
    update_graph_min_max();
    mainwindow->redrawGL();
}

void TimelineDockWidget::on_spinBox_FirstFrame_valueChanged ( int d ){
	int new_min = d;
	dock->spinBox_LastFrame->setMinimum(new_min+1);

	if(position_graph){
		position_graph->min_frame = new_min;
		update_graph_min_max();
		mainwindow->redrawGL();
	}
}

void TimelineDockWidget::on_spinBox_LastFrame_valueChanged ( int d ){
	int new_max = d;
	dock->spinBox_FirstFrame->setMaximum(new_max-1);

	if(position_graph){
		position_graph->max_frame = new_max;
		update_graph_min_max();
		mainwindow->redrawGL();
	}
}



void TimelineDockWidget::getValues(double *xyzypr){
	xyzypr[0] = dock->doubleSpinBox_X->value();
    xyzypr[1] = dock->doubleSpinBox_Y->value();
    xyzypr[2] = dock->doubleSpinBox_Z->value();
	xyzypr[3] = dock->doubleSpinBox_Yaw->value();
    xyzypr[4] = dock->doubleSpinBox_Pitch->value();
	xyzypr[5] = dock->doubleSpinBox_Roll->value();
}

void TimelineDockWidget::setValues(double *xyzypr){
	dock->doubleSpinBox_X->setValue(xyzypr[0]);
	dock->doubleSpinBox_Y->setValue(xyzypr[1]);
	dock->doubleSpinBox_Z->setValue(xyzypr[2]);
	dock->doubleSpinBox_Yaw->setValue(xyzypr[3]);
	dock->doubleSpinBox_Pitch->setValue(xyzypr[4]);
	dock->doubleSpinBox_Roll->setValue(xyzypr[5]);
}

void TimelineDockWidget::setValuesEnabled(bool enabled){
	dock->doubleSpinBox_X->setEnabled(enabled);
	dock->doubleSpinBox_Y->setEnabled(enabled);
	dock->doubleSpinBox_Z->setEnabled(enabled);
	dock->doubleSpinBox_Yaw->setEnabled(enabled);
	dock->doubleSpinBox_Pitch->setEnabled(enabled);
	dock->doubleSpinBox_Roll->setEnabled(enabled);
}


void TimelineDockWidget::update_graph_min_max(int frame)
{
	if (!mainwindow->getTracker()->trial()->getXCurve(-1))
		return;

	if (!mainwindow->getTracker()->trial() || mainwindow->getTracker()->trial()->getXCurve(-1)->empty()) {
		position_graph->max_value = 180.0;
        position_graph->min_value = -180.0;
    }
    // If a frame is specified then only check that frame for a new minimum and
    // maximum.
    else if (frame != -1) {
        if (position_graph->show_x) {
			float x_value = (*mainwindow->getTracker()->trial()->getXCurve(-1))(frame);
			if (x_value > position_graph->max_value) {
                position_graph->max_value = x_value;
            }
            if (x_value < position_graph->min_value) {
                position_graph->min_value = x_value;
            }
        }
        if (position_graph->show_y) {
			float y_value = (*mainwindow->getTracker()->trial()->getYCurve(-1))(frame);
			if (y_value > position_graph->max_value) {
                position_graph->max_value = y_value;
            }
            if (y_value < position_graph->min_value) {
                position_graph->min_value = y_value;
            }
        }
        if (position_graph->show_z) {
			float z_value = (*mainwindow->getTracker()->trial()->getZCurve(-1))(frame);
			if (z_value > position_graph->max_value) {
                position_graph->max_value = z_value;
            }
            if (z_value < position_graph->min_value) {
                position_graph->min_value = z_value;
            }
        }
        if (position_graph->show_yaw) {
			float yaw_value = (*mainwindow->getTracker()->trial()->getYawCurve(-1))(frame);
			if (yaw_value > position_graph->max_value) {
                position_graph->max_value = yaw_value;
            }
            if (yaw_value < position_graph->min_value) {
                position_graph->min_value = yaw_value;
            }
        }
        if (position_graph->show_pitch) {
			float pitch_value = (*mainwindow->getTracker()->trial()->getPitchCurve(-1))(frame);
			if (pitch_value > position_graph->max_value) {
                position_graph->max_value = pitch_value;
            }
            if (pitch_value < position_graph->min_value) {
                position_graph->min_value = pitch_value;
            }
        }
        if (position_graph->show_roll) {
			float roll_value = (*mainwindow->getTracker()->trial()->getRollCurve(-1))(frame);
			if (roll_value > position_graph->max_value) {
                position_graph->max_value = roll_value;
            }
            if (roll_value < position_graph->min_value) {
                position_graph->min_value = roll_value;
            }
        }
    }
    // Otherwise we need to check all the frames.
    else {

        position_graph->min_value = 1e6;
        position_graph->max_value = -1e6;

        if (position_graph->show_x) {
            for (frame = floor(position_graph->min_frame);
                 frame < position_graph->max_frame;
                 frame += 1.0f) {
				float x_value = (*mainwindow->getTracker()->trial()->getXCurve(-1))(frame);
				if (x_value > position_graph->max_value) {
                    position_graph->max_value = x_value;
                }
                if (x_value < position_graph->min_value) {
                    position_graph->min_value = x_value;
                }
            }
        }
        if (position_graph->show_y) {
            for (frame = floor(position_graph->min_frame);
                 frame < position_graph->max_frame;
                 frame += 1.0f) {
				float y_value = (*mainwindow->getTracker()->trial()->getYCurve(-1))(frame);
				if (y_value > position_graph->max_value) {
                    position_graph->max_value = y_value;
                }
                if (y_value < position_graph->min_value) {
                    position_graph->min_value = y_value;
                }
            }
        }
        if (position_graph->show_z) {
            for (frame = floor(position_graph->min_frame);
                 frame < position_graph->max_frame;
                 frame += 1.0f) {
				float z_value = (*mainwindow->getTracker()->trial()->getZCurve(-1))(frame);
				if (z_value > position_graph->max_value) {
                    position_graph->max_value = z_value;
                }
                if (z_value < position_graph->min_value) {
                    position_graph->min_value = z_value;
                }
            }
        }
        if (position_graph->show_yaw) {
            for (frame = floor(position_graph->min_frame);
                 frame < position_graph->max_frame;
                 frame += 1.0f) {
				float yaw_value = (*mainwindow->getTracker()->trial()->getYawCurve(-1))(frame);
				if (yaw_value > position_graph->max_value) {
                    position_graph->max_value = yaw_value;
                }
                if (yaw_value < position_graph->min_value) {
                    position_graph->min_value = yaw_value;
                }
            }
        }
        if (position_graph->show_pitch) {
            for (frame = floor(position_graph->min_frame);
                 frame < position_graph->max_frame;
                 frame += 1.0f) {
				float pitch_value = (*mainwindow->getTracker()->trial()->getPitchCurve(-1))(frame);
				if (pitch_value > position_graph->max_value) {
                    position_graph->max_value = pitch_value;
                }
                if (pitch_value < position_graph->min_value) {
                    position_graph->min_value = pitch_value;
                }
            }
        }
        if (position_graph->show_roll) {
            for (frame = floor(position_graph->min_frame);
                 frame < position_graph->max_frame;
                 frame += 1.0f) {
				float roll_value = (*mainwindow->getTracker()->trial()->getRollCurve(-1))(frame);
				if (roll_value > position_graph->max_value) {
                    position_graph->max_value = roll_value;
                }
                if (roll_value < position_graph->min_value) {
                    position_graph->min_value = roll_value;
                }
            }
        }

        position_graph->min_value -= 1.0;
        position_graph->max_value += 1.0;
    }
}