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

/// \file AutoscoperMainWindow.cpp
/// \author Benjamin Knorlein, Andy Loomis

#include "ui/AutoscoperMainWindow.h"
#include "ui_AutoscoperMainWindow.h"
#include <QGridLayout>
#include "ui/FilterDockWidget.h"
#include "ui/CameraViewWidget.h"
#include "ui/TimelineDockWidget.h"
#include "ui/TrackingOptionsDialog.h"
#include "ui/GLTracker.h"
#include "ui/ImportExportTrackingOptionsDialog.h"
#include "ui_ImportExportTrackingOptionsDialog.h"
#include "ui_TrackingOptionsDialog.h"
#include "ui/OpenCLPlatformSelectDialog.h"
#include "Manip3D.hpp"
#include "ui/WorldViewWindow.h"
#include "ui/VolumeDockWidget.h"
#include "ui/NewTrialDialog.h"
#include "ui_NewTrialDialog.h"

#include "Trial.hpp"
#include "View.hpp"
#include "Tracker.hpp"
#include "CoordFrame.hpp"

#include <QSplitter>
#include <QInputDialog>
#include <QList>
#include <QFileDialog>
#include <QGLContext>
#include <QMessageBox>
#include <QShortcut>
#include <QXmlStreamWriter>

#ifndef WITH_CUDA
#include <gpu/opencl/OpenCL.hpp>
#endif

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

AutoscoperMainWindow::AutoscoperMainWindow(bool skipGpuDevice, QWidget *parent) :
												QMainWindow(parent),
												ui(new Ui::AutoscoperMainWindow), background_threshold_(-1.0){
	
	//Setup UI
	ui->setupUi(this);

	//Init Tracker and get SharedGLContext
	tracker = new Tracker();
	gltracker = new GLTracker(tracker,NULL);
	//shared_glcontext = gltracker->getSharedContext();
	
	//History
	history = new History(10);
	first_undo = true;

	//Init empty trial
	trial_filename = "";
	is_trial_saved = true;
	is_tracking_saved = true;

	setDockNestingEnabled(true);

	filters_widget =  new FilterDockWidget(this);
	this->addDockWidget(Qt::LeftDockWidgetArea, filters_widget);

	volumes_widget = new VolumeDockWidget(this);
	this->addDockWidget(Qt::LeftDockWidgetArea, volumes_widget);
	this->setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);

	// Init for currentFrame and folder directory
	curFrame = 0;
	setLastFolder(QDir::currentPath());


#ifndef WITH_CUDA
	if (!skipGpuDevice){
		OpenCLPlatformSelectDialog * dialog = new OpenCLPlatformSelectDialog(this);
		if (dialog->getNumberPlatforms() > 1)dialog->exec();
		else{ xromm::gpu::setUsedPlatform(0); }
		delete dialog;
}
#endif

	worldview = new WorldViewWindow(this);
	addDockWidget(Qt::TopDockWidgetArea, worldview);
	worldview->setFloating(true);
	worldview->hide();
	
	timeline_widget = new TimelineDockWidget(this);
	this->addDockWidget(Qt::BottomDockWidgetArea, timeline_widget, Qt::Horizontal);
	resizeDocks({ timeline_widget}, { 300 }, Qt::Horizontal);
	tracking_dialog = NULL;

	setupShortcuts();
}

AutoscoperMainWindow::~AutoscoperMainWindow(){
	delete ui;
	delete filters_widget;

	delete tracker;
	for (int i = 0; i < manipulator.size(); i++){
		delete manipulator[i];
	}
	manipulator.clear();

	delete history;
	if(tracking_dialog) {
		tracking_dialog->hide();
		delete tracking_dialog;
	}

	for (int i = 0 ; i < cameraViews.size();i++){
		delete cameraViews[i];
	}
	cameraViews.clear();
}

void AutoscoperMainWindow::closeEvent(QCloseEvent *event)
 {
	 save_trial_prompt();
     save_tracking_prompt();
     QMainWindow::closeEvent(event);
 }

GraphData* AutoscoperMainWindow::getPosition_graph(){
	return timeline_widget->getPosition_graph();
}

Manip3D * AutoscoperMainWindow::getManipulator(int idx){
	if (idx < manipulator.size() &&
		idx >= 0){
		return manipulator[idx];
	}
	else if (getTracker()->trial()->num_frames > 0){
		return manipulator[getTracker()->trial()->current_volume];
	}
	else{
		return NULL;
	}
}

void AutoscoperMainWindow::update_graph_min_max(int frame){
	timeline_widget->update_graph_min_max(frame);
}

void AutoscoperMainWindow::relayoutCameras(int rows){
	//clear Old Splitters
	for (int i = 0 ; i < cameraViews.size();i++){
		cameraViews[i]->setParent(NULL);
	}

	QObjectList objs = ui->frameWindows->children();
	for(int x = 0; x < objs.size(); x ++){
		QObjectList objs2 = objs[x]->children();
		QSplitter * vertsplit = dynamic_cast<QSplitter*> (objs[x]);
		for(int y = objs2.size() - 1 ; y >= 0; y --){	
			QSplitter * horsplit = dynamic_cast<QSplitter*> (objs2[y]);
			if(horsplit){
				QObjectList objs3 = horsplit->children();
				delete horsplit;
			}
		}	
		
		if(vertsplit){
			ui->gridLayout->removeWidget(vertsplit);
			delete vertsplit;
		}
	}
	//Create New
	cameraViewArrangement = QSize(ceil( ((double) cameraViews.size())/rows),rows);

	QSplitter * splitter = new QSplitter(this);
    splitter->setOrientation(Qt::Vertical);
	for(int i = 0; i < cameraViewArrangement.height() ; i ++){
		QSplitter * splitterHorizontal = new QSplitter(splitter);
		splitterHorizontal->setOrientation(Qt::Horizontal);
		splitter->addWidget(splitterHorizontal);
	}

	int freeSpaces = cameraViewArrangement.height() * cameraViewArrangement.width() - cameraViews.size();
	
	int count = 0;
	for (int i = 0 ; i < cameraViews.size();i++, count++){
		if(cameraViews.size() < i + freeSpaces) count++;
		QObject* obj = splitter->children().at(rows - 1 - count / (cameraViewArrangement.width()));
		QSplitter * horsplit = dynamic_cast<QSplitter*> (obj);
		if(horsplit)horsplit->addWidget(cameraViews[i]);
	}

	for(int i = 0; i < cameraViewArrangement.height() ; i ++){
		QObject* obj = splitter->children().at(i);
		QSplitter * horsplit = dynamic_cast<QSplitter*> (obj);
		if(horsplit){
			QList<int> sizelist = horsplit->sizes();
			for(int m = 0; m < sizelist.size(); m++){
					sizelist[m] = 1;
			}
			horsplit->setSizes(sizelist);
		}
	}


	ui->gridLayout->addWidget(splitter, 0, 0, 1, 1);
}

void AutoscoperMainWindow::timelineSetValue(int value){
	tracker->trial()->frame = value;
	curFrame = value;
    frame_changed();
}

int AutoscoperMainWindow::getCurrentFrame()
{
	curFrame = tracker->trial()->frame;
	return curFrame;
}

QString AutoscoperMainWindow::getLastFolder()
{
	return lastFolderPath;
}

void AutoscoperMainWindow::frame_changed()
{
    // Lock or unlock the position
	if (timeline_widget->getPosition_graph()->frame_locks.at(tracker->trial()->frame)) {       
		timeline_widget->setValuesEnabled(false);
    }
    else {
       timeline_widget->setValuesEnabled(true);
    }

    update_xyzypr_and_coord_frame();

    for (unsigned int i = 0; i < tracker->trial()->cameras.size(); ++i) {
        tracker->trial()->videos.at(i).set_frame(tracker->trial()->frame);
        tracker->view(i)->radRenderer()->set_rad(
            tracker->trial()->videos.at(i).data(),
            tracker->trial()->videos.at(i).width(),
            tracker->trial()->videos.at(i).height(),
            tracker->trial()->videos.at(i).bps());

		//((QGLContext*) shared_glcontext)->makeCurrent();
        glBindTexture(GL_TEXTURE_2D,textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     1,
                     tracker->trial()->videos.at(i).width(),
                     tracker->trial()->videos.at(i).height(),
                     0,
                     GL_LUMINANCE,
                    (tracker->trial()->videos.at(i).bps() == 8? GL_UNSIGNED_BYTE:
                                                               GL_UNSIGNED_SHORT),
                     tracker->trial()->videos.at(i).data());
        glBindTexture(GL_TEXTURE_2D,0);
    }

	redrawGL();
	QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void AutoscoperMainWindow::volume_changed()
{
	
	// Lock or unlock the position
	if (timeline_widget->getPosition_graph()->frame_locks.at(tracker->trial()->frame)) {
		timeline_widget->setValuesEnabled(false);
	}
	else {
		timeline_widget->setValuesEnabled(true);
	}

	if (getManipulator(-1))
		getManipulator(-1)->set_movePivot(ui->toolButtonMovePivot->isChecked());

	//update_xyzypr_and_coord_frame();
	timeline_widget->getSelectedNodes()->clear();
	update_graph_min_max(timeline_widget->getPosition_graph());

	redrawGL();
	QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void AutoscoperMainWindow::update_xyzypr()
{
	if (getManipulator(-1) && tracker->trial()->getVolumeMatrix(-1)){
		double xyzypr[6];
		(CoordFrame::from_matrix(trans(getManipulator(-1)->transform())) * *tracker->trial()->getVolumeMatrix(-1)).to_xyzypr(xyzypr);

		////Update the spin buttons.
		timeline_widget->setSpinButtonUpdate(false);
		timeline_widget->setValues(&xyzypr[0]);
		timeline_widget->setSpinButtonUpdate(true);
	}
	redrawGL();
}

// Updates the coordinate frames position after the spin buttons values have
// been changed.
void AutoscoperMainWindow::update_xyzypr_and_coord_frame()
{
	for (int i = 0; i < tracker->trial()->num_volumes; i++){
		if (tracker->trial()->getXCurve(i)->empty()) {
			continue;
		}

		double xyzypr[6];
		xyzypr[0] = (*tracker->trial()->getXCurve(i))(tracker->trial()->frame);
		xyzypr[1] = (*tracker->trial()->getYCurve(i))(tracker->trial()->frame);
		xyzypr[2] = (*tracker->trial()->getZCurve(i))(tracker->trial()->frame);
		xyzypr[3] = (*tracker->trial()->getYawCurve(i))(tracker->trial()->frame);
		xyzypr[4] = (*tracker->trial()->getPitchCurve(i))(tracker->trial()->frame);
		xyzypr[5] = (*tracker->trial()->getRollCurve(i))(tracker->trial()->frame);

		CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
		set_manip_matrix(i, newCoordFrame*tracker->trial()->getVolumeMatrix(i)->inverse());

		if (i == tracker->trial()->current_volume){
			timeline_widget->setSpinButtonUpdate(false);
			timeline_widget->setValues(&xyzypr[0]);
			timeline_widget->setSpinButtonUpdate(true);
		}
	}
}

void AutoscoperMainWindow::set_manip_matrix(int idx, const CoordFrame& frame)
{
    double m[16];
    frame.to_matrix_row_order(m);
	getManipulator(idx)->set_transform(Mat4d(m));
}

// Automatically updates the graph's minimum and maximum values to stretch the
// data the full height of the viewport.
void AutoscoperMainWindow::update_graph_min_max(GraphData* graph, int frame)
{
	if (!tracker->trial() || tracker->trial()->getXCurve(-1)->empty()) {
        graph->max_value = 180.0;
        graph->min_value = -180.0;
    }
    // If a frame is specified then only check that frame for a new minimum and
    // maximum.
    else if (frame != -1) {
        if (graph->show_x) {
			float x_value = (*tracker->trial()->getXCurve(-1))(frame);
            if (x_value > graph->max_value) {
                graph->max_value = x_value;
            }
            if (x_value < graph->min_value) {
                graph->min_value = x_value;
            }
        }
        if (graph->show_y) {
			float y_value = (*tracker->trial()->getYCurve(-1))(frame);
            if (y_value > graph->max_value) {
                graph->max_value = y_value;
            }
            if (y_value < graph->min_value) {
                graph->min_value = y_value;
            }
        }
        if (graph->show_z) {
			float z_value = (*tracker->trial()->getZCurve(-1))(frame);
            if (z_value > graph->max_value) {
                graph->max_value = z_value;
            }
            if (z_value < graph->min_value) {
                graph->min_value = z_value;
            }
        }
        if (graph->show_yaw) {
			float yaw_value = (*tracker->trial()->getYawCurve(-1))(frame);
            if (yaw_value > graph->max_value) {
                graph->max_value = yaw_value;
            }
            if (yaw_value < graph->min_value) {
                graph->min_value = yaw_value;
            }
        }
        if (graph->show_pitch) {
			float pitch_value = (*tracker->trial()->getPitchCurve(-1))(frame);
            if (pitch_value > graph->max_value) {
                graph->max_value = pitch_value;
            }
            if (pitch_value < graph->min_value) {
                graph->min_value = pitch_value;
            }
        }
        if (graph->show_roll) {
			float roll_value = (*tracker->trial()->getRollCurve(-1))(frame);
            if (roll_value > graph->max_value) {
                graph->max_value = roll_value;
            }
            if (roll_value < graph->min_value) {
                graph->min_value = roll_value;
            }
        }
    }
    // Otherwise we need to check all the frames.
    else {

        graph->min_value = 1e6;
        graph->max_value = -1e6;

        if (graph->show_x) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
				float x_value = (*tracker->trial()->getXCurve(-1))(frame);
                if (x_value > graph->max_value) {
                    graph->max_value = x_value;
                }
                if (x_value < graph->min_value) {
                    graph->min_value = x_value;
                }
            }
        }
        if (graph->show_y) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
				float y_value = (*tracker->trial()->getYCurve(-1))(frame);
                if (y_value > graph->max_value) {
                    graph->max_value = y_value;
                }
                if (y_value < graph->min_value) {
                    graph->min_value = y_value;
                }
            }
        }
        if (graph->show_z) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
				float z_value = (*tracker->trial()->getZCurve(-1))(frame);
                if (z_value > graph->max_value) {
                    graph->max_value = z_value;
                }
                if (z_value < graph->min_value) {
                    graph->min_value = z_value;
                }
            }
        }
        if (graph->show_yaw) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
				float yaw_value = (*tracker->trial()->getYawCurve(-1))(frame);
                if (yaw_value > graph->max_value) {
                    graph->max_value = yaw_value;
                }
                if (yaw_value < graph->min_value) {
                    graph->min_value = yaw_value;
                }
            }
        }
        if (graph->show_pitch) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
				float pitch_value = (*tracker->trial()->getPitchCurve(-1))(frame);
                if (pitch_value > graph->max_value) {
                    graph->max_value = pitch_value;
                }
                if (pitch_value < graph->min_value) {
                    graph->min_value = pitch_value;
                }
            }
        }
        if (graph->show_roll) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
				float roll_value = (*tracker->trial()->getRollCurve(-1))(frame);
                if (roll_value > graph->max_value) {
                    graph->max_value = roll_value;
                }
                if (roll_value < graph->min_value) {
                    graph->min_value = roll_value;
                }
            }
        }

        graph->min_value -= 1.0;
        graph->max_value += 1.0;
    }
}

void AutoscoperMainWindow::setupUI()
{
    //Remove previous cameras
    for (unsigned int i = 0; i < cameraViews.size(); i++) {
		cameraViews[i]->setParent(NULL);
		delete cameraViews[i];
    }
	cameraViews.clear();
	filters_widget->clearTree();
	volumes_widget->clear();

	//Add Volumes
	for (unsigned int i = 0; i < tracker->trial()->volumes.size(); i++) {
		volumes_widget->addVolume(tracker->trial()->volumes[i].name());
	}

    //Add the new cameras
    for (unsigned int i = 0; i < tracker->trial()->cameras.size(); i++) {
		cameraViews.push_back(new CameraViewWidget(i, tracker->view(i),tracker->trial()->cameras[i].mayacam().c_str(), this));
		//cameraViews[i]->setSharedGLContext(shared_glcontext);	
		filters_widget->addCamera(tracker->view(i));
    }
	relayoutCameras(1);
    textures.resize(tracker->trial()->cameras.size());
	//QOpenGLContext::globalShareContext()->makeCurrent();
    for (unsigned i = 0; i < textures.size(); i++) {
        glGenTextures(1,&textures[i]);
        glBindTexture(GL_TEXTURE_2D,textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     1,
                     tracker->trial()->videos.at(i).width(),
                     tracker->trial()->videos.at(i).height(),
                     0,
                     GL_LUMINANCE,
                    (tracker->trial()->videos.at(i).bps() == 8? GL_UNSIGNED_BYTE:
                                                               GL_UNSIGNED_SHORT),
                     tracker->trial()->videos.at(i).data());
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D,0);
    }
	// Setup the default view

    // Update the number of frames
	timeline_widget->setFramesRange(0,tracker->trial()->num_frames-1);
    reset_graph();

    // Update the coordinate frames
	timeline_widget->getPosition_graph()->min_frame = 0;
	timeline_widget->getPosition_graph()->max_frame = tracker->trial()->num_frames-1; //
    timeline_widget->getPosition_graph()->frame_locks = vector<bool>(tracker->trial()->num_frames,false);

    update_graph_min_max(timeline_widget->getPosition_graph());

    frame_changed();
}

void AutoscoperMainWindow::redrawGL(){
	for (unsigned int i = 0; i < cameraViews.size(); i++) {
		cameraViews[i]->draw();
    }
	worldview->draw();
	timeline_widget->draw();
}

void AutoscoperMainWindow::setFrame(int frame){
	tracker->trial()->frame = frame;
	timeline_widget->setFrame(frame);
    frame_changed();
}

void AutoscoperMainWindow::update_coord_frame()
{
    double xyzypr[6];
	timeline_widget->getValues(&xyzypr[0]);
	if (tracker->trial()) {
		CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
		set_manip_matrix(tracker->trial()->current_volume, newCoordFrame*tracker->trial()->getVolumeMatrix(-1)->inverse());
	}
	redrawGL();
}


//Saving and Loading
void AutoscoperMainWindow::save_trial_prompt()
{
    if (is_trial_saved) {
        return;
    }

	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(this, "", "Would you like to save the current trial?",
                                QMessageBox::Yes|QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		//on_save_trial1_activate(NULL,NULL);
	}
}

void AutoscoperMainWindow::save_tracking_prompt()
{
    if (is_tracking_saved) {
        return;
    }

	QMessageBox::StandardButton reply;
	reply = QMessageBox::question(this, "", "Would you like to export the unsaved tracking data?",
                                QMessageBox::Yes|QMessageBox::No);
	if (reply == QMessageBox::Yes) {
		QString filename = get_filename(true, "*.tra");
		save_tracking_results(filename);
	}
}

QString AutoscoperMainWindow::get_filename(bool save, QString type)
{	
	QString FileName = "";
	QString new_filename = getLastFolder();

	if(save){
		//FileName = QFileDialog::getSaveFileName(this,
		//							tr("Save File as"), QDir::currentPath(),tr("CFG Files (") + type + tr(" *.cfg)"));
		FileName = QFileDialog::getSaveFileName(this,
									tr("Save File as"), new_filename,tr("CFG Files (") + type + tr(" *.cfg)"));

	}else{
		//FileName = QFileDialog::getOpenFileName(this,
		//							tr("Open File"), QDir::currentPath(),tr("CFG Files (") + type + tr(" *.cfg)"));
		FileName = QFileDialog::getOpenFileName(this,
									tr("Open File"), new_filename,tr("CFG Files (") + type + tr(" *.cfg)"));
	}

	// Save new last directory
	QFileInfo fi(FileName);
	setLastFolder(fi.absoluteFilePath());

	return FileName;
}


void AutoscoperMainWindow::setLastFolder(QString lastFolder)
{
	lastFolderPath = lastFolder;
}



void AutoscoperMainWindow::save_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume){
	const char* s = save_with_commas ? "," : " ";

	std::ofstream file(filename.toStdString(), ios::out);

	int start, stop;
	if (volume == -1)
	{
		start = 0;
		stop = tracker->trial()->num_volumes;
	}
	else
	{
		start = volume;
		stop = volume + 1;
	}

	file.precision(16);
	file.setf(ios::fixed, ios::floatfield);
	bool invalid;
	for (int i = 0; i < tracker->trial()->num_frames; ++i) {
		for (int j = start; j < stop; j++){
			if (!interpolate) {
				if (tracker->trial()->getXCurve(-1)->find(i) ==
					tracker->trial()->getXCurve(-1)->end() &&
					tracker->trial()->getYCurve(-1)->find(i) ==
					tracker->trial()->getYCurve(-1)->end() &&
					tracker->trial()->getZCurve(-1)->find(i) ==
					tracker->trial()->getZCurve(-1)->end() &&
					tracker->trial()->getYawCurve(-1)->find(i) ==
					tracker->trial()->getYawCurve(-1)->end() &&
					tracker->trial()->getPitchCurve(-1)->find(i) ==
					tracker->trial()->getPitchCurve(-1)->end() &&
					tracker->trial()->getRollCurve(-1)->find(i) ==
					tracker->trial()->getRollCurve(-1)->end()) {
					invalid = true;
				}
				else{
					invalid = false;
				}
			}
			else{
				invalid = false;
			}

			if (invalid){
				if (save_as_matrix) {
					file << "NaN";
					for (int j = 0; j < 15; j++) { file << s << "NaN"; }
				}
				else {
					file << "NaN";
					for (int j = 0; j < 5; j++) { file << s << "NaN"; }
				}
			}
			else{
				double xyzypr[6];
				xyzypr[0] = (*tracker->trial()->getXCurve(j))(i);
				xyzypr[1] = (*tracker->trial()->getYCurve(j))(i);
				xyzypr[2] = (*tracker->trial()->getZCurve(j))(i);
				xyzypr[3] = (*tracker->trial()->getYawCurve(j))(i);
				xyzypr[4] = (*tracker->trial()->getPitchCurve(j))(i);
				xyzypr[5] = (*tracker->trial()->getRollCurve(j))(i);

				if (save_as_matrix) {
					double m[16];
					CoordFrame::from_xyzypr(xyzypr).to_matrix(m);

					if (convert_to_cm) {
						m[12] /= 10.0;
						m[13] /= 10.0;
						m[14] /= 10.0;
					}

					if (save_as_rows) {
						file << m[0] << s << m[4] << s << m[8] << s << m[12] << s
							<< m[1] << s << m[5] << s << m[9] << s << m[13] << s
							<< m[2] << s << m[6] << s << m[10] << s << m[14] << s
							<< m[3] << s << m[7] << s << m[11] << s << m[15];
					}
					else {
						file << m[0] << s << m[1] << s << m[2] << s << m[3] << s
							<< m[4] << s << m[5] << s << m[6] << s << m[7] << s
							<< m[8] << s << m[9] << s << m[10] << s << m[11] << s
							<< m[12] << s << m[13] << s << m[14] << s << m[15];
					}
				}
				else {
					if (convert_to_cm) {
						xyzypr[0] /= 10.0;
						xyzypr[1] /= 10.0;
						xyzypr[2] /= 10.0;
					}
					if (convert_to_rad) {
						xyzypr[3] *= M_PI / 180.0;
						xyzypr[4] *= M_PI / 180.0;
						xyzypr[5] *= M_PI / 180.0;
					}

					file << xyzypr[0] << s << xyzypr[1] << s << xyzypr[2] << s
						<< xyzypr[3] << s << xyzypr[4] << s << xyzypr[5];
				}
			}

			if (j != tracker->trial()->num_volumes - 1) file << s;
		}
		file << endl;
	}
	file.close();
}

void AutoscoperMainWindow::loadFilterSettings(int camera, QString filename)
{
	filters_widget->loadFilterSettings(camera, filename);
}

std::vector<double> AutoscoperMainWindow::getPose(unsigned int volume, unsigned int frame)
{
	std::vector<double> pose(6,0);
	pose[0] = (*tracker->trial()->getXCurve(volume))(frame);
	pose[1] = (*tracker->trial()->getYCurve(volume))(frame);
	pose[2] = (*tracker->trial()->getZCurve(volume))(frame);
	pose[3] = (*tracker->trial()->getYawCurve(volume))(frame);
	pose[4] = (*tracker->trial()->getPitchCurve(volume))(frame);
	pose[5] = (*tracker->trial()->getRollCurve(volume))(frame);
	return pose;
}

void AutoscoperMainWindow::setPose(std::vector<double> pose, unsigned volume, unsigned frame)
{
	tracker->trial()->getXCurve(volume)->insert(frame, pose[0]);
	tracker->trial()->getYCurve(volume)->insert(frame, pose[1]);
	tracker->trial()->getZCurve(volume)->insert(frame, pose[2]);
	tracker->trial()->getYawCurve(volume)->insert(frame, pose[3]);
	tracker->trial()->getPitchCurve(volume)->insert(frame, pose[4]);
	tracker->trial()->getRollCurve(volume)->insert(frame, pose[5]);
}

void AutoscoperMainWindow::setBackground(double threshold)
{
	if (background_threshold_ < 0){
		for (xromm::Video &vi : tracker->trial()->videos)
			vi.create_background_image();

		tracker->updateBackground();
	}

	background_threshold_ = threshold;
	tracker->setBackgroundThreshold(background_threshold_);
}

std::vector<double> AutoscoperMainWindow::getNCC(unsigned volumeID, double* xyzpr)
{
	return tracker->trackFrame(volumeID, xyzpr);
}

std::vector<unsigned char> AutoscoperMainWindow::getImageData(unsigned volumeID, unsigned camera, double* xyzpr, unsigned& width, unsigned& height)
{
	return tracker->getImageData(volumeID, camera, xyzpr, width, height);
}

void AutoscoperMainWindow::save_tracking_results(QString filename)
{
	ImportExportTrackingOptionsDialog * diag = new ImportExportTrackingOptionsDialog(this);
	diag->exec();

	if(diag->result()){
		bool save_as_matrix = diag->diag->radioButton_TypeMatrix->isChecked();
		bool save_as_rows = diag->diag->radioButton_OrientationRow->isChecked();
		bool save_with_commas = diag->diag->radioButton_SeperatorComma->isChecked();
		bool convert_to_cm = diag->diag->radioButton_TranslationCM->isChecked();
		bool convert_to_rad = diag->diag->radioButton_RotationRadians->isChecked();
		bool interpolate = diag->diag->radioButton_InterpolationSpline->isChecked();		
		int volume = -1;
		if (diag->diag->radioButton_VolumeCurrent->isChecked()) volume = tracker->trial()->current_volume;
		save_tracking_results(filename, save_as_matrix,save_as_rows,save_with_commas,convert_to_cm,convert_to_rad,interpolate, volume);

		is_tracking_saved = true;
	}
	delete diag;
}



void AutoscoperMainWindow::backup_tracking(bool backup_on)
{
	int volume = -1;
	volume = tracker->trial()->current_volume;
	QString tracking_filename_out = "tracked_model_backup.tra";
	//save_tracking_results(tracking_filename_out, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate, volume);
	save_tracking_results(tracking_filename_out, 1, 1, 1, 0, 0, 0, volume);
}



void AutoscoperMainWindow::load_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume){
	char s = save_with_commas ? ',' : ' ';

	int start, stop;
	if (volume == -1)
	{
		start = 0;
		stop = tracker->trial()->num_volumes;
	}
	else
	{
		start = volume;
		stop = volume + 1;
	}
	std::ifstream file(filename.toStdString().c_str(), ios::in);

	for (int j = start; j < stop; j++){
		tracker->trial()->getXCurve(j)->clear();
		tracker->trial()->getYCurve(j)->clear();
		tracker->trial()->getZCurve(j)->clear();
		tracker->trial()->getYawCurve(j)->clear();
		tracker->trial()->getPitchCurve(j)->clear();
		tracker->trial()->getRollCurve(j)->clear();
	}

	double m[16];
	string line, value;
	for (int i = 0; i < tracker->trial()->num_frames && getline(file, line); ++i) {
		istringstream lineStream(line);
		for (int k = start; k < stop; k++){
			for (int j = 0; j < (save_as_matrix ? 16 : 6) && getline(lineStream, value, s); ++j) {
				istringstream valStream(value);
				valStream >> m[j];
			}

			if (value.compare(0, 3, "NaN") == 0) {
				continue;
			}

			if (save_as_matrix && save_as_rows) {
				double n[16];
				memcpy(n, m, 16 * sizeof(double));
				m[1] = n[4];
				m[2] = n[8];
				m[3] = n[12];
				m[4] = n[1];
				m[6] = n[9];
				m[7] = n[13];
				m[8] = n[2];
				m[9] = n[6];
				m[11] = n[14];
				m[12] = n[3];
				m[13] = n[7];
				m[14] = n[11];
			}

			if (convert_to_cm) {
				if (save_as_matrix) {
					m[12] *= 10.0;
					m[13] *= 10.0;
					m[14] *= 10.0;
				}
				else {
					m[0] *= 10.0;
					m[1] *= 10.0;
					m[2] *= 10.0;
				}
			}

			if (convert_to_rad) {
				if (!save_as_matrix) {
					m[3] *= 180.0 / M_PI;
					m[4] *= 180.0 / M_PI;
					m[5] *= 180.0 / M_PI;
				}
			}

			if (save_as_matrix) {
				CoordFrame::from_matrix(m).to_xyzypr(m);
			}

			tracker->trial()->getXCurve(k)->insert(i, m[0]);
			tracker->trial()->getYCurve(k)->insert(i, m[1]);
			tracker->trial()->getZCurve(k)->insert(i, m[2]);
			tracker->trial()->getYawCurve(k)->insert(i, m[3]);
			tracker->trial()->getPitchCurve(k)->insert(i, m[4]);
			tracker->trial()->getRollCurve(k)->insert(i, m[5]);
		}
	}
	file.close();

	is_tracking_saved = true;

	frame_changed();
	update_graph_min_max(timeline_widget->getPosition_graph());

	redrawGL();
}

void AutoscoperMainWindow::load_tracking_results(QString filename)
{
    save_tracking_prompt();

	ImportExportTrackingOptionsDialog * diag = new ImportExportTrackingOptionsDialog(this);
	diag->exec();

	if(diag->result()){
		bool save_as_matrix = diag->diag->radioButton_TypeMatrix->isChecked();
		bool save_as_rows = diag->diag->radioButton_OrientationRow->isChecked();
		bool save_with_commas = diag->diag->radioButton_SeperatorComma->isChecked();
		bool convert_to_cm = diag->diag->radioButton_TranslationCM->isChecked();
		bool convert_to_rad = diag->diag->radioButton_RotationRadians->isChecked();
		bool interpolate = diag->diag->radioButton_InterpolationSpline->isChecked();
		int volume = -1;
		if (diag->diag->radioButton_VolumeCurrent->isChecked()) volume = tracker->trial()->current_volume;

		load_tracking_results(filename, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate, volume);
	}
	delete diag;
}

void AutoscoperMainWindow::openTrial(){
	QString cfg_fileName = get_filename(false);

	if ( cfg_fileName.isNull() == false )
    {
        openTrial(cfg_fileName);
	}
}

void AutoscoperMainWindow::openTrial(QString filename){
    try {
		Trial * trial = new Trial(filename.toStdString());
		tracker->load(*trial);
		delete trial;

		trial_filename = filename.toStdString();
		std:cerr << "Filename: " << trial_filename << endl;
		is_trial_saved = true;
		is_tracking_saved = true;

		for (int i = 0; i < manipulator.size(); i++){
			delete manipulator[i];
		}
		manipulator.clear();
		for (int i = 0; i < tracker->trial()->num_volumes; i++){
			manipulator.push_back(new Manip3D());
			getManipulator(i)->set_transform(Mat4d());
		}

		setupUI();
		timelineSetValue(0);

		timeline_widget->setTrial(tracker->trial());

	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
}

void AutoscoperMainWindow::newTrial(){
	NewTrialDialog * diag = new NewTrialDialog(this);
	diag->exec();

	if(diag->result()){
		try {
			tracker->load(diag->trial);

			trial_filename = "";
			is_trial_saved = false;
			is_tracking_saved = true;

			for (int i = 0; i < manipulator.size(); i++){
				delete manipulator[i];
			}
			manipulator.clear();
			for (int i = 0; i < tracker->trial()->num_volumes; i++){
				manipulator.push_back(new Manip3D());
				getManipulator(i)->set_transform(Mat4d());
			}

			setupUI();
			timelineSetValue(0);

			timeline_widget->setTrial(tracker->trial());

		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
	}
	delete diag;
}

//History
void AutoscoperMainWindow::push_state()
{
    State current_state;
	for (int i = 0; i < tracker->trial()->num_volumes; i++){
		current_state.x_curve.push_back(*tracker->trial()->getXCurve(i));
		current_state.y_curve.push_back(*tracker->trial()->getYCurve(i));
		current_state.z_curve.push_back(*tracker->trial()->getZCurve(i));
		current_state.x_rot_curve.push_back(*tracker->trial()->getYawCurve(i));
		current_state.y_rot_curve.push_back(*tracker->trial()->getPitchCurve(i));
		current_state.z_rot_curve.push_back(*tracker->trial()->getRollCurve(i));
	}
    history->push(current_state);

    first_undo = true;
    is_tracking_saved = false;
}

void AutoscoperMainWindow::undo_state()
{
    if (history->can_undo()) {

        if (first_undo) {
            push_state();
            history->undo();
            first_undo = false;
        }

        State undo_state = history->undo();

		for (int i = 0; i < tracker->trial()->num_volumes; i++){
			*tracker->trial()->getXCurve(i) = undo_state.x_curve[i];
			*tracker->trial()->getYCurve(i) = undo_state.y_curve[i];
			*tracker->trial()->getZCurve(i) = undo_state.z_curve[i];
			*tracker->trial()->getYawCurve(i) = undo_state.x_rot_curve[i];
			*tracker->trial()->getPitchCurve(i) = undo_state.y_rot_curve[i];
			*tracker->trial()->getRollCurve(i) = undo_state.z_rot_curve[i];
		}

        timeline_widget->getSelectedNodes()->clear();

        update_graph_min_max(timeline_widget->getPosition_graph());
        update_xyzypr_and_coord_frame();
        
		redrawGL();
    }
}

void AutoscoperMainWindow::redo_state()
{
    if (history->can_redo()) {
        State redo_state = history->redo();

		for (int i = 0; i < tracker->trial()->num_volumes; i++){
			*tracker->trial()->getXCurve(i) = redo_state.x_curve[i];
			*tracker->trial()->getYCurve(i) = redo_state.y_curve[i];
			*tracker->trial()->getZCurve(i) = redo_state.z_curve[i];
			*tracker->trial()->getYawCurve(i) = redo_state.x_rot_curve[i];
			*tracker->trial()->getPitchCurve(i) = redo_state.y_rot_curve[i];
			*tracker->trial()->getRollCurve(i) = redo_state.z_rot_curve[i];
		}
		timeline_widget->getSelectedNodes()->clear();

        update_graph_min_max(timeline_widget->getPosition_graph());
        update_xyzypr_and_coord_frame();
        
		redrawGL();
    }
}

void AutoscoperMainWindow::reset_graph()
{
	for (int i = 0; i < tracker->trial()->num_volumes; i++){
		tracker->trial()->getXCurve(i)->clear();
		tracker->trial()->getYCurve(i)->clear();
		tracker->trial()->getZCurve(i)->clear();
		tracker->trial()->getYawCurve(i)->clear();
		tracker->trial()->getPitchCurve(i)->clear();
		tracker->trial()->getRollCurve(i)->clear();
	}

	timeline_widget->getCopiedNodes()->clear();
}

//File menu

void AutoscoperMainWindow::on_actionNew_triggered(bool checked){
	save_trial_prompt();
    save_tracking_prompt();

	newTrial();
}
void AutoscoperMainWindow::on_actionOpen_triggered(bool checked){
	save_trial_prompt();
    save_tracking_prompt();
	
	openTrial();
}

void AutoscoperMainWindow::on_actionSave_triggered(bool checked){
	if (trial_filename.compare("") == 0) {
        on_actionSave_as_triggered(true);
    }
    else {
        try {
            tracker->trial()->save(trial_filename);
            is_trial_saved = true;
        }
        catch (exception& e) {
            cerr << e.what() << endl;
        }
    }
}

void AutoscoperMainWindow::on_actionSave_as_triggered(bool checked){
	QString filename = get_filename(true);
    if (filename.compare("") != 0) {
        try {
			trial_filename = filename.toStdString();
			tracker->trial()->save(trial_filename);
            is_tracking_saved = true;
        }
        catch (exception& e) {
            cerr << e.what() << endl;
        }
    }
}

void AutoscoperMainWindow::on_actionImport_Tracking_triggered(bool checked){
	QString filename = get_filename(false, "*.tra");
    if (filename.compare("") != 0) {
        load_tracking_results(filename);
    }
}

void AutoscoperMainWindow::on_actionExport_Tracking_triggered(bool checked){
	QString filename = get_filename(true, "*.tra");
    if (filename.compare("") != 0) {
        save_tracking_results(filename);
    }
}

void AutoscoperMainWindow::on_actionQuit_triggered(bool checked){
	QApplication::quit();
}

void AutoscoperMainWindow::on_actionSave_Test_Sequence_triggered(bool checked){
	fprintf(stderr, "Saving testSequence\n");
	for (int i = 0; i < tracker->trial()->num_frames; i++){
		timeline_widget->setFrame(i);
		for (int j = 0; j < cameraViews.size(); j++){
			QFileInfo fi(cameraViews[j]->getName());
			if (i == 0){
				QDir dir(fi.absolutePath() + OS_SEP + fi.completeBaseName());
				if (!dir.exists()){
					dir.mkdir(".");
				}
			}
			QString filename = fi.absolutePath() + OS_SEP + fi.completeBaseName() + OS_SEP + fi.completeBaseName() + QString().sprintf("%05d", i) + ".pgm";
			cameraViews[j]->saveFrame(filename);
		}
		QApplication::processEvents();
	}
}

void AutoscoperMainWindow::on_actionSaveForBatch_triggered(bool checked){
	QString inputPath = QFileDialog::getExistingDirectory(this,
		tr("Select Directory"), QDir::currentPath());
	if (inputPath.isNull() == false)
	{
		QString xml_filename = inputPath + OS_SEP + "batch.xml";
		if (!xml_filename.isNull())
		{
			QFile file(xml_filename);
			if (file.open(QIODevice::WriteOnly | QIODevice::Text))
			{
				QXmlStreamWriter xmlWriter(&file);
				xmlWriter.writeStartDocument();
				xmlWriter.writeStartElement("Batch");
				xmlWriter.setAutoFormatting(true);
				//save GPU_devices
#ifndef WITH_CUDA
				xmlWriter.writeStartElement("GPUDevice");
				xmlWriter.writeAttribute("Platform", QString::number(xromm::gpu::getUsedPlatform().first));
				xmlWriter.writeAttribute("Device", QString::number(xromm::gpu::getUsedPlatform().second));
				xmlWriter.writeEndElement();
#endif
				//save Trial
				QString trial_filename = inputPath + OS_SEP + "trial.cfg";
				tracker->trial()->save(trial_filename.toStdString());
				xmlWriter.writeStartElement("Trial");
				xmlWriter.writeCharacters(trial_filename);
				xmlWriter.writeEndElement();

				//save Filters
				filters_widget->saveAllSettings(inputPath + OS_SEP);

				//save Pivot
				for (int i = 0; i < tracker->trial()->num_volumes; i++){
					xmlWriter.writeStartElement("Pivot");
					xmlWriter.writeAttribute("id", QString::number(i));
					xmlWriter.writeCharacters(QString::fromStdString(tracker->trial()->getVolumeMatrix(i)->to_string()));
					xmlWriter.writeEndElement();
				}

				//save Tracking
				ImportExportTrackingOptionsDialog * diag = new ImportExportTrackingOptionsDialog(this);
				diag->exec();

				bool save_as_matrix = diag->diag->radioButton_TypeMatrix->isChecked();
				bool save_as_rows = diag->diag->radioButton_OrientationRow->isChecked();
				bool save_with_commas = diag->diag->radioButton_SeperatorComma->isChecked();
				bool convert_to_cm = diag->diag->radioButton_TranslationCM->isChecked();
				bool convert_to_rad = diag->diag->radioButton_RotationRadians->isChecked();
				bool interpolate = diag->diag->radioButton_InterpolationSpline->isChecked();

				QString tracking_filename = inputPath + OS_SEP + "track_data.cfg";
				save_tracking_results(tracking_filename, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate);

				xmlWriter.writeStartElement("TrackingData");
				xmlWriter.writeAttribute("Matrix", QString::number(save_as_matrix));
				xmlWriter.writeAttribute("Rows", QString::number(save_as_rows));
				xmlWriter.writeAttribute("Commas", QString::number(save_with_commas));
				xmlWriter.writeAttribute("cm", QString::number(convert_to_cm));
				xmlWriter.writeAttribute("rad", QString::number(convert_to_rad));
				xmlWriter.writeAttribute("interpolate", QString::number(interpolate));
				xmlWriter.writeCharacters(tracking_filename);
				xmlWriter.writeEndElement();
				delete diag;

				//save TrackingOptions
				TrackingOptionsDialog * tracking_dialog_tmp;
				tracking_dialog_tmp = new TrackingOptionsDialog(this);
				tracking_dialog_tmp->setRange(timeline_widget->getPosition_graph()->min_frame, timeline_widget->getPosition_graph()->max_frame, tracker->trial()->num_frames - 1);
				tracking_dialog_tmp->inActive = true;
				tracking_dialog_tmp->exec();
				xmlWriter.writeStartElement("TrackingOptions");
				xmlWriter.writeAttribute("Start", QString::number(tracking_dialog_tmp->diag->spinBox_FrameStart->value()));
				xmlWriter.writeAttribute("End", QString::number(tracking_dialog_tmp->diag->spinBox_FrameEnd->value()));
				xmlWriter.writeAttribute("Guess", QString::number(getTracker()->trial()->guess));
				xmlWriter.writeAttribute("Iterations", QString::number(tracking_dialog_tmp->diag->spinBox_NumberRefinements->value()));
				xmlWriter.writeEndElement();

				xmlWriter.writeEndElement();

				delete tracking_dialog_tmp;
				xmlWriter.writeEndDocument();
				file.close();
			}
		}
	}
}

void AutoscoperMainWindow::runBatch(QString batchfile, bool saveData){
	bool save_as_matrix;
	bool save_as_rows;
	bool save_with_commas;
	bool convert_to_cm;
	bool convert_to_rad;
	bool interpolate;

	int start_Frame = 0;
	int end_Frame;
	int iterations = 1;

	bool doTracking = false;
	QString trackdata_filename;

	gltracker->makeCurrent();

	if (batchfile.isNull() == false)
	{

		QString xml_filename = batchfile;
		if (!xml_filename.isNull())
		{
			QFile file(xml_filename);
			if (file.open(QIODevice::ReadOnly | QIODevice::Text))
			{
				QXmlStreamReader xmlReader(&file);
				//Reading from the file

				while (!xmlReader.atEnd())
				{
					if (xmlReader.isStartElement())
					{
						QString name = xmlReader.name().toString();
						if (name == "GPUDevice")
						{
#ifndef WITH_CUDA
							fprintf(stderr, "Load GPUDevice Setting\n");
							xromm::gpu::setUsedPlatform(xmlReader.readElementText().toInt());
							QApplication::processEvents();
#endif
						}
						else if (name == "Trial")
						{
							fprintf(stderr, "Load Trial Setting\n");
							openTrial(xmlReader.readElementText());
							end_Frame = tracker->trial()->num_frames;
							QFileInfo info(file);
							filters_widget->loadAllSettings(info.absolutePath() + OS_SEP);
							QApplication::processEvents();
						}
						else if (name == "Pivot")
						{
							QXmlStreamAttributes attr = xmlReader.attributes();
							int id = attr.value("id").toString().toInt();
							fprintf(stderr, "Load Pivot %d Setting\n", id);
							QString pivot_data = xmlReader.readElementText();
							tracker->trial()->getVolumeMatrix(id)->from_string(pivot_data.toStdString());
						}
						else if (name == "TrackingData")
						{
							fprintf(stderr, "Load TrackingData Setting\n");
							QXmlStreamAttributes attr = xmlReader.attributes();
							save_as_matrix = attr.value("Matrix").toString().toInt();
							save_as_rows = attr.value("Rows").toString().toInt();
							save_with_commas = attr.value("Commas").toString().toInt();
							convert_to_cm = attr.value("cm").toString().toInt();
							convert_to_rad = attr.value("rad").toString().toInt();
							interpolate = attr.value("interpolate").toString().toInt();
							trackdata_filename = xmlReader.readElementText();

							load_tracking_results(trackdata_filename, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate);
							QApplication::processEvents();
						}
						else if (name == "TrackingOptions")
						{
							fprintf(stderr, "Load TrackingOptions Setting\n");
							doTracking = true;
							QXmlStreamAttributes attr = xmlReader.attributes();
							start_Frame = attr.value("Start").toString().toInt();
							end_Frame = attr.value("End").toString().toInt();
							getTracker()->trial()->guess = attr.value("Guess").toString().toInt();
							iterations = attr.value("Iterations").toString().toInt();
						}
						xmlReader.readNextStartElement();
					}
					else
					{
						xmlReader.readNext();
					}
				}
				if (xmlReader.hasError())
				{
					std::cout << "XML error: " << xmlReader.error() << std::endl;
				}
				file.close();
			}
		}
	}

	if (doTracking){
		fprintf(stderr, "Start Tracking\n");
		TrackingOptionsDialog * tracking_dialog_tmp;
		tracking_dialog_tmp = new TrackingOptionsDialog(this);
		tracking_dialog_tmp->diag->spinBox_FrameStart->setValue(start_Frame);
		tracking_dialog_tmp->diag->spinBox_FrameEnd->setValue(end_Frame);
		tracking_dialog_tmp->diag->spinBox_NumberRefinements->setValue(iterations);
		QApplication::processEvents();
		tracking_dialog_tmp->on_pushButton_OK_clicked(true);

		if (saveData){

			QFileInfo info(trackdata_filename);
			QString tracking_filename_out = info.absolutePath() + OS_SEP + info.completeBaseName() + "_tracked.tra";
			fprintf(stderr, "Save Data to %s\n", tracking_filename_out.toStdString());
			save_tracking_results(tracking_filename_out, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate);
		}

		delete tracking_dialog_tmp;
	}
}


void AutoscoperMainWindow::on_actionLoad_xml_batch_triggered(bool checked){


	QString inputfile= QFileDialog::getOpenFileName(this,
									tr("Open XML File"), QDir::currentPath(),tr("XML Files (") + tr(" *.xml)"));
	if ( inputfile.isNull() == false )
    {
		fprintf(stderr,"%s\n",inputfile.toStdString());
		runBatch(inputfile);
    }
}


//Edit menu
void AutoscoperMainWindow::on_actionUndo_triggered(bool checked){
	undo_state();
}

void AutoscoperMainWindow::on_actionRedo_triggered(bool checked){
	redo_state();
}

void AutoscoperMainWindow::on_actionCut_triggered(bool checked){
	push_state();

    if (!timeline_widget->getSelectedNodes()->empty()) {
        timeline_widget->getCopiedNodes()->clear();
        for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
            if ((*timeline_widget->getSelectedNodes())[i].second == NODE) {
				timeline_widget->getCopiedNodes()->push_back( (*timeline_widget->getSelectedNodes())[i].first);
                if (!timeline_widget->getPosition_graph()->frame_locks.at((int)(*timeline_widget->getSelectedNodes())[i].first.first->time((*timeline_widget->getSelectedNodes())[i].first.second))) {
                    (*timeline_widget->getSelectedNodes())[i].first.first->erase((*timeline_widget->getSelectedNodes())[i].first.second);
                }
            }
        }
        timeline_widget->getSelectedNodes()->clear();
    }

    update_xyzypr_and_coord_frame();
	update_graph_min_max(timeline_widget->getPosition_graph());

    redrawGL();
}

void AutoscoperMainWindow::on_actionCopy_triggered(bool checked){
	if (!timeline_widget->getSelectedNodes()->empty()) {
		timeline_widget->getCopiedNodes()->clear();
        for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
            if ((*timeline_widget->getSelectedNodes())[i].second == NODE) {
                timeline_widget->getCopiedNodes()->push_back((*timeline_widget->getSelectedNodes())[i].first);
            }
        }
    }

	redrawGL();
}

void AutoscoperMainWindow::on_actionPaste_triggered(bool checked){
	push_state();

    if (!timeline_widget->getCopiedNodes()->empty()) {
        float frame_offset = timeline_widget->getCopiedNodes()->front().first->time(timeline_widget->getCopiedNodes()->front().second);
        for (unsigned i = 0; i < timeline_widget->getCopiedNodes()->size(); i++) {
			float frame = tracker->trial()->frame + (*timeline_widget->getCopiedNodes())[i].first->time((*timeline_widget->getCopiedNodes())[i].second) - frame_offset;
			if (!timeline_widget->getPosition_graph()->frame_locks.at((int)frame)) {
				if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve::X_CURVE){
					getTracker()->trial()->getXCurve(-1)->insert(frame, (*timeline_widget->getCopiedNodes())[i].first->value((*timeline_widget->getCopiedNodes())[i].second));
				}
				else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve::Y_CURVE){
					getTracker()->trial()->getYCurve(-1)->insert(frame, (*timeline_widget->getCopiedNodes())[i].first->value((*timeline_widget->getCopiedNodes())[i].second));
				}
				else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve::Z_CURVE){
					getTracker()->trial()->getZCurve(-1)->insert(frame, (*timeline_widget->getCopiedNodes())[i].first->value((*timeline_widget->getCopiedNodes())[i].second));
				}
				else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve::YAW_CURVE){
					getTracker()->trial()->getYawCurve(-1)->insert(frame, (*timeline_widget->getCopiedNodes())[i].first->value((*timeline_widget->getCopiedNodes())[i].second));
				}
				else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve::PITCH_CURVE){
					getTracker()->trial()->getPitchCurve(-1)->insert(frame, (*timeline_widget->getCopiedNodes())[i].first->value((*timeline_widget->getCopiedNodes())[i].second));
				}
				else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve::ROLL_CURVE){
					getTracker()->trial()->getRollCurve(-1)->insert(frame, (*timeline_widget->getCopiedNodes())[i].first->value((*timeline_widget->getCopiedNodes())[i].second));
				}
			}
        }
        timeline_widget->getSelectedNodes()->clear();
    }

    update_xyzypr_and_coord_frame();
    update_graph_min_max(timeline_widget->getPosition_graph());

	redrawGL();
}

void AutoscoperMainWindow::on_actionDelete_triggered(bool checked){
	if (!timeline_widget->getSelectedNodes()->empty()) {
        push_state();

        for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
            if ((*timeline_widget->getSelectedNodes())[i].second == NODE) {
				if (!timeline_widget->getPosition_graph()->frame_locks.at((int)(*timeline_widget->getSelectedNodes())[i].first.first->time((*timeline_widget->getSelectedNodes())[i].first.second))) {
                    (*timeline_widget->getSelectedNodes())[i].first.first->erase((*timeline_widget->getSelectedNodes())[i].first.second);
                }
            }
        }
        timeline_widget->getSelectedNodes()->clear();

        update_xyzypr_and_coord_frame();
        update_graph_min_max(timeline_widget->getPosition_graph());

		redrawGL();
    }
}

void AutoscoperMainWindow::on_actionSet_Background_triggered(bool checked)
{
	bool ok;
	double threshold = QInputDialog::getDouble(this, "Enter threshold", "threshold in percent", 0.2, 0.0, 1.0, 4, &ok);
	if (ok){
		setBackground(threshold);
	}
}

//Tracking menu
void AutoscoperMainWindow::on_actionImport_triggered(bool checked){
	QString filename = get_filename(false,"*.tra");
    if (filename.compare("") != 0) {
        load_tracking_results(filename);
    }
}

void AutoscoperMainWindow::on_actionExport_triggered(bool checked){
	QString filename = get_filename(true, "*.tra");
    if (filename.compare("") != 0) {
        save_tracking_results(filename);
    }
}

void AutoscoperMainWindow::on_actionInsert_Key_triggered(bool checked){
	push_state();
	timeline_widget->getSelectedNodes()->clear();

    double xyzypr[6];
	(CoordFrame::from_matrix(trans(getManipulator()->transform()))* *tracker->trial()->getVolumeMatrix(-1)).to_xyzypr(xyzypr);
	getTracker()->trial()->getXCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[0]);
	getTracker()->trial()->getYCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[1]);
	getTracker()->trial()->getZCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[2]);
	getTracker()->trial()->getYawCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[3]);
	getTracker()->trial()->getPitchCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[4]);
	getTracker()->trial()->getRollCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[5]);

	timeline_widget->update_graph_min_max();

	redrawGL();
}

void AutoscoperMainWindow::on_actionLock_triggered(bool checked){
	push_state();

    for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {

        int time = (int)(*timeline_widget->getSelectedNodes())[i].first.first->time(
                   (*timeline_widget->getSelectedNodes())[i].first.second);

        // Force the addition of keys for all curves in order to truely freeze
        // the frame

		tracker->trial()->getXCurve(-1)->insert(time);
		tracker->trial()->getYCurve(-1)->insert(time);
		tracker->trial()->getZCurve(-1)->insert(time);
		tracker->trial()->getYawCurve(-1)->insert(time);
		tracker->trial()->getPitchCurve(-1)->insert(time);
		tracker->trial()->getRollCurve(-1)->insert(time);

        timeline_widget->getPosition_graph()->frame_locks.at(time) = true;
    }

    timeline_widget->getSelectedNodes()->clear();

    frame_changed();
    redrawGL();
}

void AutoscoperMainWindow::on_actionUnlock_triggered(bool checked){
	for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
		timeline_widget->getPosition_graph()->frame_locks.at((int)(*timeline_widget->getSelectedNodes())[i].first.first->time(
                                   (*timeline_widget->getSelectedNodes())[i].first.second)) = false;
    }

    frame_changed();
    redrawGL();
}

void AutoscoperMainWindow::on_actionBreak_Tangents_triggered(bool checked){
	push_state();

    for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
        KeyCurve& curve = *(*timeline_widget->getSelectedNodes())[i].first.first;
        KeyCurve::iterator it = (*timeline_widget->getSelectedNodes())[i].first.second;
        if (!timeline_widget->getPosition_graph()->frame_locks.at((int)curve.time(it))) {
            curve.set_bind_tangents(it,false);
        }
    }

}

void AutoscoperMainWindow::on_actionSmooth_Tangents_triggered(bool checked){
	push_state();

    for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
        KeyCurve& curve = *(*timeline_widget->getSelectedNodes())[i].first.first;
        KeyCurve::iterator it = (*timeline_widget->getSelectedNodes())[i].first.second;

        if (!timeline_widget->getPosition_graph()->frame_locks.at((int)curve.time(it))) {
            curve.set_bind_tangents(it,true);
            curve.set_in_tangent_type(it,KeyCurve::SMOOTH);
            curve.set_out_tangent_type(it,KeyCurve::SMOOTH);
        }
    }

    update_xyzypr_and_coord_frame();
    update_graph_min_max(timeline_widget->getPosition_graph());

    redrawGL();
}

//View

void AutoscoperMainWindow::on_actionLayoutCameraViews_triggered(bool triggered){
	
	bool ok;
    int rows = QInputDialog::getInt(this, tr("Layput Camera Views"),
                                          tr("Number of Rows"),1,1,10,1, &ok);
    if (ok)relayoutCameras(rows);
}

void AutoscoperMainWindow::on_actionShow_world_view_triggered(bool checked){
	if(checked){
		worldview->show();
	}else{
		worldview->hide();
	}
}

//Toolbar
void AutoscoperMainWindow::on_toolButtonOpenTrial_clicked(){
	save_trial_prompt();
    save_tracking_prompt();
	
	openTrial();
}

void AutoscoperMainWindow::on_toolButtonSaveTracking_clicked(){
	QString filename = get_filename(true, "*.tra");
    if (filename.compare("") != 0) {
        save_tracking_results(filename);
    }
}

void AutoscoperMainWindow::on_toolButtonLoadTracking_clicked(){
	QString filename = get_filename(false, "*.tra");
    if (filename.compare("") != 0) {
        load_tracking_results(filename);
    }
}

void AutoscoperMainWindow::on_toolButtonTranslate_clicked(){
	ui->toolButtonTranslate->setChecked(true);
	ui->toolButtonRotate->setChecked(false);
	getManipulator()->set_mode(Manip3D::TRANSLATION);
	redrawGL();
}

void AutoscoperMainWindow::on_toolButtonRotate_clicked(){
	ui->toolButtonRotate->setChecked(true);
	ui->toolButtonTranslate->setChecked(false);
	getManipulator()->set_mode(Manip3D::ROTATION);
	redrawGL();
}

void AutoscoperMainWindow::on_toolButtonMovePivot_clicked(){
	getManipulator()->set_movePivot(ui->toolButtonMovePivot->isChecked());
	redrawGL();
}

void AutoscoperMainWindow::on_toolButtonTrack_clicked(){
	if(tracking_dialog == NULL)
		tracking_dialog = new TrackingOptionsDialog(this);

	tracking_dialog->setRange(timeline_widget->getPosition_graph()->min_frame,timeline_widget->getPosition_graph()->max_frame, 
		tracker->trial()->num_frames-1);
	
	tracking_dialog->show();
}

void AutoscoperMainWindow::on_toolButtonTrackCurrent_clicked() {
	if (tracking_dialog == NULL)
		tracking_dialog = new TrackingOptionsDialog(this);

	tracking_dialog->setRange(timeline_widget->getPosition_graph()->min_frame, timeline_widget->getPosition_graph()->max_frame,
		tracker->trial()->num_frames - 1);

	tracking_dialog->trackCurrent();
}


void AutoscoperMainWindow::on_toolButtonRetrack_clicked(){
	if(tracking_dialog == NULL)
		tracking_dialog = new TrackingOptionsDialog(this);

	tracking_dialog->setRange(timeline_widget->getPosition_graph()->min_frame,timeline_widget->getPosition_graph()->max_frame, 
		tracker->trial()->num_frames-1);
	
	tracking_dialog->retrack();
}

void AutoscoperMainWindow::key_w_pressed(){
	ui->toolButtonTranslate->click();
}
void AutoscoperMainWindow::key_e_pressed(){
	ui->toolButtonRotate->click();
}	
void AutoscoperMainWindow::key_d_pressed(){
	ui->toolButtonMovePivot->click();
}	
void AutoscoperMainWindow::key_h_pressed(){
	filters_widget->toggle_drrs();
}
void AutoscoperMainWindow::key_t_pressed(){
	ui->toolButtonTrack->click();
}	
void AutoscoperMainWindow::key_p_pressed(){
	ui->toolButtonRetrack->click();
}	
void AutoscoperMainWindow::key_c_pressed() {
	on_actionInsert_Key_triggered(true); // Insert the current frame and then run the tracking
	ui->toolButtonTrackCurrent->click();
}
void AutoscoperMainWindow::key_plus_pressed(){
	getManipulator(-1)->set_pivotSize(getManipulator(-1)->get_pivotSize() * 1.1f);
	redrawGL();
}
void AutoscoperMainWindow::key_equal_pressed(){
	getManipulator(-1)->set_pivotSize(getManipulator(-1)->get_pivotSize() * 1.1f);
	redrawGL();
}

void AutoscoperMainWindow::key_minus_pressed(){
	getManipulator(-1)->set_pivotSize(getManipulator(-1)->get_pivotSize() * 0.9f);
	redrawGL();
}

//Shortcuts
void AutoscoperMainWindow::setupShortcuts(){
	new QShortcut(QKeySequence(Qt::Key_W), this, SLOT(key_w_pressed()));
	new QShortcut(QKeySequence(Qt::Key_E), this, SLOT(key_e_pressed()));
	new QShortcut(QKeySequence(Qt::Key_D), this, SLOT(key_d_pressed()));
	new QShortcut(QKeySequence(Qt::Key_H), this, SLOT(key_h_pressed()));
	new QShortcut(QKeySequence(Qt::Key_T), this, SLOT(key_t_pressed()));
	new QShortcut(QKeySequence(Qt::Key_P), this, SLOT(key_p_pressed()));
	new QShortcut(QKeySequence(Qt::Key_C), this, SLOT(key_c_pressed()));
	new QShortcut(QKeySequence(Qt::Key_Plus), this, SLOT(key_plus_pressed()));
	new QShortcut(QKeySequence(Qt::Key_Equal), this, SLOT(key_equal_pressed()));
	new QShortcut(QKeySequence(Qt::Key_Minus), this, SLOT(key_minus_pressed()));

	ui->actionCopy->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_C));
	ui->actionPaste->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_V));
	ui->actionCut->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_X));
	ui->actionDelete->setShortcut(QKeySequence(Qt::Key_Delete));
	ui->actionUndo->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Z));
	ui->actionRedo->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Y));

	ui->actionNew->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_N));
	ui->actionOpen->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));
	ui->actionSave->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_S));
	ui->actionSave_as->setShortcut(QKeySequence(Qt::SHIFT + Qt::CTRL + Qt::Key_S));
	ui->actionQuit->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q));
	ui->actionInsert_Key->setShortcut(QKeySequence(Qt::Key_S));
}