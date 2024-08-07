// ----------------------------------
// Copyright (c) 2011-2019, Brown University
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
#include "ui/AdvancedOptionsDialog.h"

#include "ui/GLTracker.h"
#include "ui/ImportExportTrackingOptionsDialog.h"
#include "ui_ImportExportTrackingOptionsDialog.h"
#include "ui_TrackingOptionsDialog.h"
#include "ui/AboutAutoscoper.h"
#include "ui/OpenCLPlatformSelectDialog.h"
#include "Manip3D.hpp"
#include "ui/WorldViewWindow.h"
#include "ui/VolumeDockWidget.h"
#include "ui/NewTrialDialog.h"
#include "ui_NewTrialDialog.h"

#include "asys/SystemTools.hxx"
#include "Trial.hpp"
#include "View.hpp"
#include "Tracker.hpp"
#include "CoordFrame.hpp"
#include "KeyCurve.hpp"

#include <QSplitter>
#include <QInputDialog>
#include <QList>
#include <QFileDialog>
#include <QOpenGLContext>
#include <QMessageBox>
#include <QShortcut>
#include <QXmlStreamWriter>

#if defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
#  include <gpu/opencl/OpenCL.hpp>
#endif

#include <ctime>
#include <iostream>
#include "filesystem_compat.hpp"
#include <fstream>
#include <sstream>

#ifdef WIN32
#  define OS_SEP "\\"
#else
#  define OS_SEP "/"
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

AutoscoperMainWindow::AutoscoperMainWindow(bool skipGpuDevice, QWidget* parent)
  : QMainWindow(parent)
  , ui(new Ui::AutoscoperMainWindow)
  , background_threshold_(-1.0)
{

  // Setup UI
  ui->setupUi(this);

  // VERSION NUMBER
  puts("Autoscoper v2.8\n");

  // Init Tracker and get SharedGLContext
  tracker = new Tracker();
  gltracker = new GLTracker(tracker, NULL);
  // shared_glcontext = gltracker->getSharedContext();

  // History
  history = new History(20);
  first_undo = true;

  // Init empty trial
  trial_filename = "";
  is_trial_saved = true;
  is_tracking_saved = true;

  setDockNestingEnabled(true);

  // Create filter widget and put it on the left
  filters_widget = new FilterDockWidget(this);
  this->addDockWidget(Qt::LeftDockWidgetArea, filters_widget);

  // Create volume idwget and put it on the bottom-left
  volumes_widget = new VolumeDockWidget(this);
  this->addDockWidget(Qt::LeftDockWidgetArea, volumes_widget);
  this->setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);

  // Initialize currentFrame and set the last folder directory
  curFrame = 0;
  setLastFolder(QDir::currentPath());

#if defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
  if (!skipGpuDevice) {
    OpenCLPlatformSelectDialog* dialog = new OpenCLPlatformSelectDialog(this);
    if (dialog->getNumberPlatforms() > 1)
      dialog->exec();
    else {
      xromm::gpu::setUsedPlatform(0);
    }
    delete dialog;
  }
#endif
  // Create worldview but hide it
  worldview = new WorldViewWindow(this);
  addDockWidget(Qt::TopDockWidgetArea, worldview);
  worldview->setFloating(true);
  worldview->hide();

  // Create timeline widget
  timeline_widget = new TimelineDockWidget(this);
  this->addDockWidget(Qt::BottomDockWidgetArea, timeline_widget, Qt::Horizontal);
  // resizeDocks({ timeline_widget}, { 300 }, Qt::Horizontal);

  // Initialize other dialogs/widgets
  tracking_dialog = NULL;
  about_autoscoper = NULL;
  advanced_dialog = NULL;

  // Setup Shortcuts
  setupShortcuts();

  // Enable the sample data menus if the sample data is installed
#if defined(Autoscoper_INSTALL_SAMPLE_DATA)
  ui->actionOpen_Sample_Wrist->setEnabled(true);
  ui->actionOpen_Sample_Knee->setEnabled(true);
  ui->actionOpen_Sample_Ankle->setEnabled(true);
#endif
}

AutoscoperMainWindow::~AutoscoperMainWindow()
{
  delete ui;

  delete filters_widget;
  delete volumes_widget;
  delete worldview;
  delete tracker;
  for (int i = 0; i < manipulator.size(); i++) {
    delete manipulator[i];
  }
  manipulator.clear();

  delete history;
  if (tracking_dialog) {
    tracking_dialog->hide();
    delete tracking_dialog;
  }
  if (about_autoscoper) {
    about_autoscoper->hide();
    delete about_autoscoper;
  }
  if (advanced_dialog) {
    advanced_dialog->hide();
    delete advanced_dialog;
  }

  cameraViews.clear();
  for (int i = 0; i < cameraViews.size(); i++) {
    delete cameraViews[i];
  }
}

void AutoscoperMainWindow::closeEvent(QCloseEvent* event)
{
  save_trial_prompt();
  save_tracking_prompt();
  QMainWindow::closeEvent(event);
}

GraphData* AutoscoperMainWindow::getPosition_graph()
{
  return timeline_widget->getPosition_graph();
}

Manip3D* AutoscoperMainWindow::getManipulator(int idx)
{
  if (idx < manipulator.size() && idx >= 0) {
    return manipulator[idx];
  } else if (getTracker()->trial()->num_frames > 0) {
    return manipulator[getTracker()->trial()->current_volume];
  } else {
    return NULL;
  }
}

void AutoscoperMainWindow::update_graph_min_max(int frame)
{
  timeline_widget->update_graph_min_max(frame);
}

void AutoscoperMainWindow::relayoutCameras(int rows)
{
  // clear Old Splitters
  for (int i = 0; i < cameraViews.size(); i++) {
    cameraViews[i]->setParent(NULL);
  }

  QObjectList objs = ui->frameWindows->children();
  for (int x = 0; x < objs.size(); x++) {
    QObjectList objs2 = objs[x]->children();
    QSplitter* vertsplit = dynamic_cast<QSplitter*>(objs[x]);
    for (int y = objs2.size() - 1; y >= 0; y--) {
      QSplitter* horsplit = dynamic_cast<QSplitter*>(objs2[y]);
      if (horsplit) {
        QObjectList objs3 = horsplit->children();
        delete horsplit;
      }
    }

    if (vertsplit) {
      ui->gridLayout->removeWidget(vertsplit);
      delete vertsplit;
    }
  }
  // Create New
  cameraViewArrangement = QSize(ceil(((double)cameraViews.size()) / rows), rows);

  QSplitter* splitter = new QSplitter(this);
  splitter->setOrientation(Qt::Vertical);
  for (int i = 0; i < cameraViewArrangement.height(); i++) {
    QSplitter* splitterHorizontal = new QSplitter(splitter);
    splitterHorizontal->setOrientation(Qt::Horizontal);
    splitter->addWidget(splitterHorizontal);
  }

  size_t freeSpaces = cameraViewArrangement.height() * cameraViewArrangement.width() - cameraViews.size();

  int count = 0;
  for (int i = 0; i < cameraViews.size(); i++, count++) {
    if (cameraViews.size() < i + freeSpaces)
      count++;
    QObject* obj = splitter->children().at(rows - 1 - count / (cameraViewArrangement.width()));
    QSplitter* horsplit = dynamic_cast<QSplitter*>(obj);
    if (horsplit)
      horsplit->addWidget(cameraViews[i]);
  }

  for (int i = 0; i < cameraViewArrangement.height(); i++) {
    QObject* obj = splitter->children().at(i);
    QSplitter* horsplit = dynamic_cast<QSplitter*>(obj);
    if (horsplit) {
      QList<int> sizelist = horsplit->sizes();
      for (int m = 0; m < sizelist.size(); m++) {
        sizelist[m] = 1;
      }
      horsplit->setSizes(sizelist);
    }
  }

  ui->gridLayout->addWidget(splitter, 0, 0, 1, 1);
}

void AutoscoperMainWindow::timelineSetValue(int value)
{
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
  } else {
    timeline_widget->setValuesEnabled(true);
  }

  update_xyzypr_and_coord_frame();

  for (unsigned int i = 0; i < tracker->trial()->cameras.size(); ++i) {
    tracker->trial()->videos.at(i).set_frame(tracker->trial()->frame);
    tracker->view(i)->radRenderer()->set_rad(tracker->trial()->videos.at(i).data(),
                                             tracker->trial()->videos.at(i).width(),
                                             tracker->trial()->videos.at(i).height(),
                                             tracker->trial()->videos.at(i).bps());

    // ((QOpenGLContext*) shared_glcontext)->makeCurrent();

    glBindTexture(GL_TEXTURE_2D, textures[i]);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 1,
                 tracker->trial()->videos.at(i).width(),
                 tracker->trial()->videos.at(i).height(),
                 0,
                 GL_LUMINANCE,
                 (tracker->trial()->videos.at(i).bps() == 8 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT),
                 tracker->trial()->videos.at(i).data());
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  redrawGL();
  QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void AutoscoperMainWindow::volume_changed()
{
  // Lock or unlock the position
  if (timeline_widget->getPosition_graph()->frame_locks.at(tracker->trial()->frame)) {
    timeline_widget->setValuesEnabled(false);
  } else {
    timeline_widget->setValuesEnabled(true);
  }

  if (getManipulator(-1))
    getManipulator(-1)->set_movePivot(ui->toolButtonMovePivot->isChecked());

  // update_xyzypr_and_coord_frame();
  timeline_widget->getSelectedNodes()->clear();
  update_graph_min_max(timeline_widget->getPosition_graph());
  redrawGL();
  // QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void AutoscoperMainWindow::update_xyzypr()
{
  if (getManipulator(-1) && tracker->trial()->getVolumeMatrix(-1)) {
    double xyzypr[6];
    (CoordFrame::from_matrix(trans(getManipulator(-1)->transform())) * *tracker->trial()->getVolumeMatrix(-1))
      .to_xyzypr(xyzypr);

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
  for (int i = 0; i < tracker->trial()->num_volumes; i++) {
    if (tracker->trial()->getXCurve(i)->empty()) {
      continue;
    }

    float x_val = (*tracker->trial()->getXCurve(i))(tracker->trial()->frame);
    float y_val = (*tracker->trial()->getYCurve(i))(tracker->trial()->frame);
    float z_val = (*tracker->trial()->getZCurve(i))(tracker->trial()->frame);
    Quatf quat_val = (*tracker->trial()->getQuatCurve(i))(tracker->trial()->frame);
    Vec3f eulers = quat_val.toEuler();

    double xyzypr[6] = { x_val, y_val, z_val, eulers.z, eulers.y, eulers.x };

    CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
    set_manip_matrix(i, newCoordFrame * tracker->trial()->getVolumeMatrix(i)->inverse());

    if (i == tracker->trial()->current_volume) {
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
    Vec3f eulers = (*tracker->trial()->getQuatCurve(-1))(frame).toEuler();
    if (graph->show_yaw) {
      if (eulers.z > graph->max_value) {
        graph->max_value = eulers.z;
      }
      if (eulers.z < graph->min_value) {
        graph->min_value = eulers.z;
      }
    }
    if (graph->show_pitch) {
      if (eulers.y > graph->max_value) {
        graph->max_value = eulers.y;
      }
      if (eulers.y < graph->min_value) {
        graph->min_value = eulers.y;
      }
    }
    if (graph->show_roll) {
      if (eulers.x > graph->max_value) {
        graph->max_value = eulers.x;
      }
      if (eulers.x < graph->min_value) {
        graph->min_value = eulers.x;
      }
    }
  }
  // Otherwise we need to check all the frames.
  else {

    graph->min_value = 1e6;
    graph->max_value = -1e6;

    double min_max[2] = { 1e6, -1e6 };

    for (frame = floor(graph->min_frame); frame < graph->max_frame; frame += 1.0f) {
      if (graph->show_x) {
        float x_value = (*tracker->trial()->getXCurve(-1))(frame);
        if (x_value > min_max[1])
          min_max[1] = x_value;
        if (x_value < min_max[0])
          min_max[0] = x_value;
      }
      if (graph->show_y) {
        float y_value = (*tracker->trial()->getYCurve(-1))(frame);
        if (y_value > min_max[1])
          min_max[1] = y_value;
        if (y_value < min_max[0])
          min_max[0] = y_value;
      }
      if (graph->show_z) {
        float z_value = (*tracker->trial()->getZCurve(-1))(frame);
        if (z_value > min_max[1])
          min_max[1] = z_value;
        if (z_value < min_max[0])
          min_max[0] = z_value;
      }
      Vec3f eulers = (*tracker->trial()->getQuatCurve(-1))(frame).toEuler();
      if (graph->show_yaw) {
        if (eulers.z > min_max[1])
          min_max[1] = eulers.z;
        if (eulers.z < min_max[0])
          min_max[0] = eulers.z;
      }
      if (graph->show_pitch) {
        if (eulers.y > min_max[1])
          min_max[1] = eulers.y;
        if (eulers.y < min_max[0])
          min_max[0] = eulers.y;
      }
      if (graph->show_roll) {
        if (eulers.x > min_max[1])
          min_max[1] = eulers.x;
        if (eulers.x < min_max[0])
          min_max[0] = eulers.x;
      }
    }

    graph->min_value = min_max[0];
    graph->max_value = min_max[1];
  }
}

void AutoscoperMainWindow::setupUI()
{
  cameraViews.erase(cameraViews.begin(), cameraViews.end());

  for (unsigned int i = 0; i < cameraViews.size(); i++) {
    cameraViews[i]->setParent(NULL);
    delete cameraViews[i];
  }

  cameraViews.clear();
  filters_widget->clearTree();
  volumes_widget->clearVol();

  for (int i = 0; i < tracker->trial()->num_volumes; i++) {
    manipulator.push_back(new Manip3D());
    getManipulator(i)->set_transform(Mat4d());
  }

  // Add Volumes
  for (unsigned int i = 0; i < tracker->trial()->volumes.size(); i++) {
    std::cout << "Volume Name: " << tracker->trial()->volumes[i].name() << std::endl;
    volumes_widget->addVolume(tracker->trial()->volumes[i].name(), i);
  }

  // Add the new cameras
  for (unsigned int i = 0; i < tracker->trial()->cameras.size(); i++) {
    cameraViews.push_back(
      new CameraViewWidget(i, tracker->view(i), tracker->trial()->cameras[i].mayacam().c_str(), this));
    filters_widget->addCamera(tracker->view(i));
  }

  relayoutCameras(1);

  // QOpenGLContext::globalShareContext()->makeCurrent();
  // Showing the texture for 3D Worldview
  textures.resize(tracker->trial()->cameras.size());
  for (unsigned i = 0; i < textures.size(); i++) {
    glGenTextures(1, &textures[i]);
    glBindTexture(GL_TEXTURE_2D, textures[i]);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 1,
                 tracker->trial()->videos.at(i).width(),
                 tracker->trial()->videos.at(i).height(),
                 0,
                 GL_LUMINANCE,
                 (tracker->trial()->videos.at(i).bps() == 8 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT),
                 tracker->trial()->videos.at(i).data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  // Setup the default view

  // Update the number of frames
  timeline_widget->setFramesRange(0, tracker->trial()->num_frames - 1);
  reset_graph();

  // Update the coordinate frames
  timeline_widget->getPosition_graph()->min_frame = 0;
  timeline_widget->getPosition_graph()->max_frame = tracker->trial()->num_frames - 1; //
  timeline_widget->getPosition_graph()->frame_locks = std::vector<bool>(tracker->trial()->num_frames, false);

  update_graph_min_max(timeline_widget->getPosition_graph());

  frame_changed();
}

void AutoscoperMainWindow::redrawGL()
{
  for (unsigned int i = 0; i < cameraViews.size(); i++) {
    cameraViews[i]->draw();
  }
  worldview->draw();
  timeline_widget->draw();
}

void AutoscoperMainWindow::setFrame(int frame)
{
  tracker->trial()->frame = frame;
  timeline_widget->setFrame(frame);
  frame_changed();
}

void AutoscoperMainWindow::optimizeFrame(int volumeID,
                                         int frame,
                                         int dframe,
                                         int repeats,
                                         int opt_method,
                                         unsigned int max_iter,
                                         double min_limit,
                                         double max_limit,
                                         int cf_model,
                                         unsigned int stall_iter)
{

  tracker->trial()->current_volume = volumeID;

  volume_changed();

  tracker->optimize(frame, dframe, repeats, opt_method, max_iter, min_limit, max_limit, cf_model, stall_iter);
}

void AutoscoperMainWindow::update_coord_frame()
{
  double xyzypr[6];
  timeline_widget->getValues(&xyzypr[0]);
  if (tracker->trial()) {
    CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
    set_manip_matrix(tracker->trial()->current_volume,
                     newCoordFrame * tracker->trial()->getVolumeMatrix(-1)->inverse());
  }
  redrawGL();
}

// Saving and Loading
void AutoscoperMainWindow::save_trial_prompt()
{
  if (is_trial_saved) {
    return;
  }

  QMessageBox::StandardButton reply;
  reply =
    QMessageBox::question(this, "", "Would you like to save the current trial?", QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    // on_save_trial1_activate(NULL,NULL);
    on_actionSave_as_triggered(true);
  }
}

void AutoscoperMainWindow::save_tracking_prompt()
{
  if (is_tracking_saved) {
    return;
  }

  QMessageBox::StandardButton reply;
  reply = QMessageBox::question(
    this, "", "Would you like to export the unsaved tracking data?", QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    QString filename = get_filename(true, "*.tra");
    save_tracking_results(filename);
  }
}

QString AutoscoperMainWindow::get_filename(bool save, QString type)
{
  QString FileName = "";
  QString new_filename = getLastFolder();

  if (save) {
    // FileName = QFileDialog::getSaveFileName(this,
    //              tr("Save File as"), QDir::currentPath(),tr("CFG Files (") + type + tr(" *.cfg)"));
    FileName =
      QFileDialog::getSaveFileName(this, tr("Save File as"), new_filename, tr("CFG Files (") + type + tr(" *.cfg)"));

  } else {
    // FileName = QFileDialog::getOpenFileName(this,
    //              tr("Open File"), QDir::currentPath(),tr("CFG Files (") + type + tr(" *.cfg)"));
    FileName =
      QFileDialog::getOpenFileName(this, tr("Open File"), new_filename, tr("CFG Files (") + type + tr(" *.cfg)"));
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

void AutoscoperMainWindow::save_tracking_results(QString filename,
                                                 bool save_as_matrix,
                                                 bool save_as_rows,
                                                 bool save_with_commas,
                                                 bool convert_to_cm,
                                                 bool convert_to_rad,
                                                 bool interpolate,
                                                 int volume)
{
  const char* s = save_with_commas ? "," : " ";

  std::ofstream file(filename.toStdString().c_str(), std::ios::out);

  int start, stop;
  if (volume == -1) {
    start = 0;
    stop = tracker->trial()->num_volumes;
  } else {
    start = volume;
    stop = volume + 1;
  }

  file.precision(16);
  file.setf(std::ios::fixed, std::ios::floatfield);
  bool invalid;
  for (int i = 0; i < tracker->trial()->num_frames; ++i) {
    for (int j = start; j < stop; j++) {
      if (!interpolate) {
        if (tracker->trial()->getXCurve(j)->find(i) == tracker->trial()->getXCurve(j)->end()
            && tracker->trial()->getYCurve(j)->find(i) == tracker->trial()->getYCurve(j)->end()
            && tracker->trial()->getZCurve(j)->find(i) == tracker->trial()->getZCurve(j)->end()
            && tracker->trial()->getQuatCurve(j)->find(i) == tracker->trial()->getQuatCurve(j)->end()) {
          invalid = true;
        } else {
          invalid = false;
        }
      } else {
        invalid = false;
      }

      if (invalid) {
        if (save_as_matrix) {
          file << "NaN";
          for (int k = 0; k < 15; k++) {
            file << s << "NaN";
          }
        } else {
          file << "NaN";
          for (int k = 0; k < 5; k++) {
            file << s << "NaN";
          }
        }
      } else {
        double xyzypr[6];
        xyzypr[0] = (*tracker->trial()->getXCurve(j))(i);
        xyzypr[1] = (*tracker->trial()->getYCurve(j))(i);
        xyzypr[2] = (*tracker->trial()->getZCurve(j))(i);
        Vec3f eulers = (*tracker->trial()->getQuatCurve(j))(i).toEuler();
        xyzypr[3] = eulers.z;
        xyzypr[4] = eulers.y;
        xyzypr[5] = eulers.x;

        if (save_as_matrix) {
          double m[16];
          CoordFrame::from_xyzypr(xyzypr).to_matrix(m);

          if (convert_to_cm) {
            m[12] /= 10.0;
            m[13] /= 10.0;
            m[14] /= 10.0;
          }

          if (save_as_rows) {
            file << m[0] << s << m[4] << s << m[8] << s << m[12] << s << m[1] << s << m[5] << s << m[9] << s << m[13]
                 << s << m[2] << s << m[6] << s << m[10] << s << m[14] << s << m[3] << s << m[7] << s << m[11] << s
                 << m[15];
          } else {
            file << m[0] << s << m[1] << s << m[2] << s << m[3] << s << m[4] << s << m[5] << s << m[6] << s << m[7] << s
                 << m[8] << s << m[9] << s << m[10] << s << m[11] << s << m[12] << s << m[13] << s << m[14] << s
                 << m[15];
          }
        } else {
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

          file << xyzypr[0] << s << xyzypr[1] << s << xyzypr[2] << s << xyzypr[3] << s << xyzypr[4] << s << xyzypr[5];
        }
      }

      if (j != stop - 1)
        file << s;
    }
    file << std::endl;
  }
  file.close();
}

void AutoscoperMainWindow::loadFilterSettings(int camera, QString filename)
{
  // std::cout << "Test Load Filter: " << filename.toStdString() << std::endl;
  filters_widget->loadFilterSettings(camera, filename);
}

std::vector<double> AutoscoperMainWindow::getPose(unsigned int volume, unsigned int frame)
{
  if (frame == -1) {
    frame = tracker->trial()->frame;
  }

  std::vector<double> pose(6, 0);
  pose[0] = (*tracker->trial()->getXCurve(volume))(frame);
  pose[1] = (*tracker->trial()->getYCurve(volume))(frame);
  pose[2] = (*tracker->trial()->getZCurve(volume))(frame);
  Quatf q = (*tracker->trial()->getQuatCurve(volume))(frame);
  Vec3f euler = q.toEuler();
  pose[3] = euler.z;
  pose[4] = euler.y;
  pose[5] = euler.x;
  return pose;
}

void AutoscoperMainWindow::setPose(std::vector<double> pose, unsigned int volume, unsigned int frame)
{
  if (frame == -1) {
    frame = tracker->trial()->frame;
  }

  tracker->trial()->getXCurve(volume)->insert(frame, pose[0]);
  tracker->trial()->getYCurve(volume)->insert(frame, pose[1]);
  tracker->trial()->getZCurve(volume)->insert(frame, pose[2]);
  tracker->trial()->getQuatCurve(volume)->insert(frame, Quatf(pose[3], pose[4], pose[5]));
  frame_changed();
  redrawGL();
}

void AutoscoperMainWindow::setBackground(double threshold)
{
  if (background_threshold_ < 0) {
    for (xromm::Video& vi : tracker->trial()->videos) {
      if (!vi.create_background_image()) {
        std::cerr << "Error creating background image for video " << vi.dirname() << "\n"
                  << "Failed to set background threshold" << std::endl;
        return;
      }
    }

    tracker->updateBackground();
  }

  background_threshold_ = threshold;
  tracker->setBackgroundThreshold(background_threshold_);
}

std::vector<double> AutoscoperMainWindow::getNCC(unsigned int volumeID, double* xyzpr)
{
  return tracker->trackFrame(volumeID, xyzpr);
}

void AutoscoperMainWindow::saveFullDRR()
{
  /*QString filename = get_filename(true);
     if (filename.compare("") != 0) {
     try {
      std::string drr_folderpath = filename.toStdString().c_str();
     }
     catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
     }
     }*/
  getTracker()->getFullDRR(tracker->trial()->current_volume);
}

std::vector<unsigned char> AutoscoperMainWindow::getImageData(unsigned int volumeID,
                                                              unsigned int camera,
                                                              double* xyzpr,
                                                              unsigned int& width,
                                                              unsigned int& height)
{
  return tracker->getImageData(volumeID, camera, xyzpr, width, height);
}

void AutoscoperMainWindow::save_tracking_results(QString filename)
{
  ImportExportTrackingOptionsDialog* diag = new ImportExportTrackingOptionsDialog(this);
  diag->exec();

  if (diag->result()) {
    bool save_as_matrix = diag->diag->radioButton_TypeMatrix->isChecked();
    bool save_as_rows = diag->diag->radioButton_OrientationRow->isChecked();
    bool save_with_commas = diag->diag->radioButton_SeperatorComma->isChecked();
    bool convert_to_cm = diag->diag->radioButton_TranslationCM->isChecked();
    bool convert_to_rad = diag->diag->radioButton_RotationRadians->isChecked();
    bool interpolate = diag->diag->radioButton_InterpolationSpline->isChecked();
    int volume = -1;
    if (diag->diag->radioButton_VolumeCurrent->isChecked())
      volume = tracker->trial()->current_volume;
    logTrackingParametersToFile();
    save_tracking_results(
      filename, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate, volume);
    is_tracking_saved = true;
  }
  delete diag;
}

void AutoscoperMainWindow::logTrackingParametersToFile()
{
  // Create logs directory and set up filename
  QDir defaultRootDir(default_root_path);
  defaultRootDir.mkpath("Logs");
  QString filename = defaultRootDir.filePath(QString("Logs/%1_%2.log").arg(std::time(nullptr)).arg(default_task_name));
  std::cout << "Exporting log file to: " << filename.toStdString() << std::endl;

  std::ofstream file(filename.toStdString(), std::ios::out);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename.toStdString() << " for writing." << std::endl;
    return;
  }

  file << "Task Name: " << default_task_name.toStdString() << "\n";
  file << "\n";

  tracker->printLatestOptimizationParameters(file);
  file << "\n";

  filters_widget->printAllSettings(file);
  file << "\n";

  file.close();
}

void AutoscoperMainWindow::backup_tracking(bool backup_on)
{
  int volume = -1;
  volume = tracker->trial()->current_volume;
  QString tracking_filename_out = "tracked_model_backup.tra";
  // save_tracking_results(tracking_filename_out, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm,
  // convert_to_rad, interpolate, volume);
  save_tracking_results(tracking_filename_out, 1, 1, 1, 0, 0, 0, volume);
}

bool AutoscoperMainWindow::load_tracking_results(QString filename,
                                                 bool save_as_matrix,
                                                 bool save_as_rows,
                                                 bool save_with_commas,
                                                 bool convert_to_cm,
                                                 bool convert_to_rad,
                                                 bool interpolate,
                                                 int volume)
{

  if (!std::filesystem::exists(filename.toStdString())) {
    std::cerr << "Tracking Data File Not Found: " << qPrintable(filename) << std::endl;
    return false;
  }

  std::cout << "Load Tracking Data Volume " << volume << " : " << qPrintable(filename) << std::endl;
  std::cout << " save as matrix   : " << save_as_matrix
            << "\n"
               " save as rows     : "
            << save_as_rows
            << "\n"
               " save with commas : "
            << save_with_commas
            << "\n"
               " convert to cm    : "
            << convert_to_cm
            << "\n"
               " convert to rad   : "
            << convert_to_rad
            << "\n"
               " interpolate      : "
            << interpolate << std::endl;

  char s = save_with_commas ? ',' : ' ';

  int start, stop;
  if (volume == -1) {
    start = 0;
    stop = tracker->trial()->num_volumes;
  } else {
    start = volume;
    stop = volume + 1;
  }
  std::ifstream file(filename.toStdString().c_str(), std::ios::in);

  for (int j = start; j < stop; j++) {
    tracker->trial()->getXCurve(j)->clear();
    tracker->trial()->getYCurve(j)->clear();
    tracker->trial()->getZCurve(j)->clear();
    tracker->trial()->getQuatCurve(j)->clear();
  }

  double m[16];
  std::string line, value;
  for (int i = 0; i < tracker->trial()->num_frames && asys::SystemTools::GetLineFromStream(file, line); ++i) {
    std::istringstream lineStream(line);
    for (int k = start; k < stop; k++) {
      for (int j = 0; j < (save_as_matrix ? 16 : 6) && std::getline(lineStream, value, s); ++j) {
        std::istringstream valStream(value);
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
        } else {
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
      tracker->trial()->getQuatCurve(k)->insert(i, Quatf(m[3], m[4], m[5]));
    }
  }
  file.close();

  is_tracking_saved = true;

  frame_changed();
  update_graph_min_max(timeline_widget->getPosition_graph());

  redrawGL();

  return true;
}

void AutoscoperMainWindow::load_tracking_results(QString filename)
{
  save_tracking_prompt();

  ImportExportTrackingOptionsDialog* diag = new ImportExportTrackingOptionsDialog(this);
  diag->exec();

  if (diag->result()) {
    bool save_as_matrix = diag->diag->radioButton_TypeMatrix->isChecked();
    bool save_as_rows = diag->diag->radioButton_OrientationRow->isChecked();
    bool save_with_commas = diag->diag->radioButton_SeperatorComma->isChecked();
    bool convert_to_cm = diag->diag->radioButton_TranslationCM->isChecked();
    bool convert_to_rad = diag->diag->radioButton_RotationRadians->isChecked();
    bool interpolate = diag->diag->radioButton_InterpolationSpline->isChecked();
    int volume = -1;
    if (diag->diag->radioButton_VolumeCurrent->isChecked())
      volume = tracker->trial()->current_volume;

    load_tracking_results(
      filename, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate, volume);
  }
  delete diag;
}

void AutoscoperMainWindow::openTrial()
{
  QString cfg_fileName = get_filename(false);

  if (cfg_fileName.isNull() == false) {
    openTrial(cfg_fileName);
  }
}

bool AutoscoperMainWindow::openTrial(QString filename)
{
  try {
    Trial* trial = new Trial(filename.toStdString().c_str());
    tracker->load(*trial);
    delete trial;

    trial_filename = filename.toStdString().c_str();
    puts("\n\nSetting up a new trial...");
    std::cout << "Filename: " << trial_filename << std::endl;
    is_trial_saved = true;
    is_tracking_saved = true;

    for (int i = 0; i < manipulator.size(); i++) {
      delete manipulator[i];
    }
    manipulator.clear();

    setupUI();

    timelineSetValue(0);

    timeline_widget->setTrial(tracker->trial());

    // Store filename as a default for filter and saving
    // This has to change based on operating system
    size_t pos = trial_filename.find_last_of("/");
    std::string def_root_path_temp = trial_filename.substr(0, pos);
    default_root_path = QString::fromStdString(def_root_path_temp);
    std::string test = default_root_path.toStdString().c_str();

    std::cout << "Root Path is: " << test << std::endl;

    // Store Default Values:
    default_filter_folder = "xParameters";
    default_filter_name = "control_settings";
    default_tracking_folder = "Tracking";

    /////// FILTER PATH:
    QString filter_path = default_root_path;
    filter_path += "/";
    filter_path += default_filter_folder;
    filter_path += "/";
    filter_path += default_filter_name;
    filter_path += ".vie";

    // std::cerr << "Filter Path is: " << filter_path.toStdString().c_str() << std::endl;

    for (int j = 0; j < cameraViews.size(); j++) {
      loadFilterSettings(j, filter_path);
    }

    /////// TRACKED PATH:
    size_t pos_trck = trial_filename.find(".cfg");
    std::string task_name_tmp = trial_filename.substr(pos + 1, pos_trck - pos - 1);
    std::cout << "Task Name: " << task_name_tmp << std::endl;

    default_task_name = QString::fromStdString(task_name_tmp);

    QString tracking_path_root = default_root_path;
    QString tracking_path;

    std::cout << "Tracking Data Directory: " << qPrintable(tracking_path_root) << "/"
              << qPrintable(default_tracking_folder) << std::endl;

    for (int iVol = 0; iVol < tracker->trial()->num_volumes; iVol++) {

      QString tracking_filename =
        QString("%1_%2.tra").arg(QString::fromStdString(task_name_tmp), volumes_widget->getVolumeName(iVol));

      tracking_path = tracking_path_root;
      tracking_path += "/";
      tracking_path += default_tracking_folder;
      tracking_path += "/";
      tracking_path += tracking_filename;

      std::cout << "Looking for Tracking Data File: " << qPrintable(tracking_filename) << std::endl;
      if (!std::filesystem::exists(tracking_path.toStdString())) {
        std::cout << "  Not Found" << std::endl;
        continue;
      }

      load_tracking_results(tracking_path, 1, default_saving_format, 1, 0, 0, 0, iVol);
    }

    on_actionInsert_Key_triggered(true);
    // on_actionDelete_triggered(true);
    //
    return true;
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
}

void AutoscoperMainWindow::newTrial()
{
  NewTrialDialog* diag = new NewTrialDialog(this);
  diag->exec();

  if (diag->result()) {
    try {
      tracker->load(diag->trial);

      trial_filename = "";
      is_trial_saved = false;
      is_tracking_saved = true;

      for (int i = 0; i < manipulator.size(); i++) {
        delete manipulator[i];
      }
      manipulator.clear();
      for (int i = 0; i < tracker->trial()->num_volumes; i++) {
        manipulator.push_back(new Manip3D());
        getManipulator(i)->set_transform(Mat4d());
      }

      setupUI();
      timelineSetValue(0);

      timeline_widget->setTrial(tracker->trial());

      on_actionInsert_Key_triggered(true);
      on_actionDelete_triggered(true);
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }
  delete diag;
}

// History
void AutoscoperMainWindow::push_state()
{
  State current_state;
  for (int i = 0; i < tracker->trial()->num_volumes; i++) {
    current_state.x_curve.push_back(*tracker->trial()->getXCurve(i));
    current_state.y_curve.push_back(*tracker->trial()->getYCurve(i));
    current_state.z_curve.push_back(*tracker->trial()->getZCurve(i));
    current_state.quat_curve.push_back(*tracker->trial()->getQuatCurve(i));
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

    for (int i = 0; i < tracker->trial()->num_volumes; i++) {

      *tracker->trial()->getXCurve(i) = undo_state.x_curve[i];
      *tracker->trial()->getYCurve(i) = undo_state.y_curve[i];
      *tracker->trial()->getZCurve(i) = undo_state.z_curve[i];
      *tracker->trial()->getQuatCurve(i) = undo_state.quat_curve[i];
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

    for (int i = 0; i < tracker->trial()->num_volumes; i++) {
      *tracker->trial()->getXCurve(i) = redo_state.x_curve[i];
      *tracker->trial()->getYCurve(i) = redo_state.y_curve[i];
      *tracker->trial()->getZCurve(i) = redo_state.z_curve[i];
      *tracker->trial()->getQuatCurve(i) = redo_state.quat_curve[i];
    }
    timeline_widget->getSelectedNodes()->clear();

    update_graph_min_max(timeline_widget->getPosition_graph());
    update_xyzypr_and_coord_frame();

    redrawGL();
  }
}

void AutoscoperMainWindow::reset_graph()
{
  for (int i = 0; i < tracker->trial()->num_volumes; i++) {
    tracker->trial()->getXCurve(i)->clear();
    tracker->trial()->getYCurve(i)->clear();
    tracker->trial()->getZCurve(i)->clear();
    tracker->trial()->getQuatCurve(i)->clear();
  }

  timeline_widget->getCopiedNodes()->clear();
}

void AutoscoperMainWindow::MovingAverageFilter(int nWin, int firstFrame, int lastFrame)
{
  unsigned int current_volume = tracker->trial()->current_volume;

  std::vector<double> cur_pose(6, 0);
  std::vector<double> sma(6, 0);
  std::vector<double> temp_sma(6, 0);
  std::vector<double> filt_pose(6, 0);
  int nFrames = lastFrame; // tracker->trial()->num_frames;

  // Hyperparameters
  int skip_frame = 1;

  int q = (nWin - 1) / 2;
  double q_step = 1;
  for (int iFrame = firstFrame + q; iFrame <= nFrames - q; iFrame++) {
    sma.assign(6, 0);
    temp_sma.assign(6, 0);
    filt_pose.assign(6, 0);
    for (int jFrame = -q; jFrame <= q; jFrame++) {
      cur_pose = getPose(current_volume, iFrame + jFrame);

      if (jFrame == -q || jFrame == +q) {
        q_step = 1 / (4 * (double)q);
      } else {
        q_step = 1 / (2 * (double)q);
      }
      std::transform(cur_pose.begin(),
                     cur_pose.end(),
                     temp_sma.begin(),
                     std::bind(std::multiplies<double>(), q_step, std::placeholders::_1));

      std::transform(temp_sma.begin(), temp_sma.end(), filt_pose.begin(), filt_pose.begin(), std::plus<double>());
    }

    if (fmod(iFrame, skip_frame) == 0) {
      setPose(filt_pose, current_volume, iFrame);
    } else {
      deletePose(iFrame);
    }
  }
}

void AutoscoperMainWindow::deletePose(int curFrame)
{

  std::vector<double> temp_pose(6, 0);
  int current_volume = tracker->trial()->current_volume;
  setPose(temp_pose, current_volume, curFrame);

  tracker->trial()->getXCurve(-1)->erase(tracker->trial()->getXCurve(-1)->find(curFrame));
  tracker->trial()->getYCurve(-1)->erase(tracker->trial()->getYCurve(-1)->find(curFrame));
  tracker->trial()->getZCurve(-1)->erase(tracker->trial()->getZCurve(-1)->find(curFrame));
  tracker->trial()->getQuatCurve(-1)->erase(tracker->trial()->getQuatCurve(-1)->find(curFrame));
}

// File menu
void AutoscoperMainWindow::on_actionNew_triggered(bool checked)
{
  save_trial_prompt();
  save_tracking_prompt();

  newTrial();
}
void AutoscoperMainWindow::on_actionOpen_triggered(bool checked)
{
  save_trial_prompt();
  save_tracking_prompt();

  openTrial();
}

void AutoscoperMainWindow::on_actionSave_triggered(bool checked)
{
  /*if (trial_filename.compare("") == 0) {
        on_actionSave_as_triggered(true);
     }
     else {
        try {
            tracker->trial()->save(trial_filename);
            is_trial_saved = true;
        }
        catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
     }*/
  QString filename = get_filename(true, "*.tra");
  if (filename.compare("") != 0) {
    save_tracking_results(filename);
    is_tracking_saved = true;
  }
}

void AutoscoperMainWindow::on_actionSave_as_triggered(bool checked)
{
  QString filename = get_filename(true);
  if (filename.compare("") != 0) {
    try {
      trial_filename = filename.toStdString().c_str();
      tracker->trial()->save(trial_filename);
      is_trial_saved = true;
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }
}

void AutoscoperMainWindow::on_actionImport_Tracking_triggered(bool checked)
{
  QString filename = get_filename(false, "*.tra");
  if (filename.compare("") != 0) {
    load_tracking_results(filename);
  }
}

void AutoscoperMainWindow::on_actionExport_Tracking_triggered(bool checked)
{
  QString filename = get_filename(true, "*.tra");
  if (filename.compare("") != 0) {
    save_tracking_results(filename);
  }
}

void AutoscoperMainWindow::on_actionQuit_triggered(bool checked)
{
  QApplication::quit();
}

void AutoscoperMainWindow::on_actionSave_Test_Sequence_triggered(bool checked)
{
  fprintf(stderr, "Saving testSequence\n");
  for (int i = 0; i < tracker->trial()->num_frames; i++) {
    timeline_widget->setFrame(i);
    for (int j = 0; j < cameraViews.size(); j++) {
      QFileInfo fi(cameraViews[j]->getName());
      if (i == 0) {
        QDir dir(fi.absolutePath() + OS_SEP + fi.completeBaseName());
        if (!dir.exists()) {
          dir.mkdir(".");
        }
      }
#if (QT_VERSION >= QT_VERSION_CHECK(5, 5, 0))
      QString filename = fi.absolutePath() + OS_SEP + fi.completeBaseName() + OS_SEP + fi.completeBaseName()
                         + QString().asprintf("%05d", i) + ".pgm";
#else
      QString filename = fi.absolutePath() + OS_SEP + fi.completeBaseName() + OS_SEP + fi.completeBaseName()
                         + QString().sprintf("%05d", i) + ".pgm";
#endif
      cameraViews[j]->saveFrame(filename);
    }
    QApplication::processEvents();
  }
}

void AutoscoperMainWindow::on_actionSaveForBatch_triggered(bool checked)
{
  QString inputPath = QFileDialog::getExistingDirectory(this, tr("Select Directory"), QDir::currentPath());
  if (inputPath.isNull() == false) {
    QString xml_filename = inputPath + OS_SEP + "batch.xml";
    if (!xml_filename.isNull()) {
      QFile file(xml_filename);
      if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QXmlStreamWriter xmlWriter(&file);
        xmlWriter.writeStartDocument();
        xmlWriter.writeStartElement("Batch");
        xmlWriter.setAutoFormatting(true);
        // save GPU_devices
#if defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
        xmlWriter.writeStartElement("GPUDevice");
        xmlWriter.writeAttribute("Platform", QString::number(xromm::gpu::getUsedPlatform().first));
        xmlWriter.writeAttribute("Device", QString::number(xromm::gpu::getUsedPlatform().second));
        xmlWriter.writeEndElement();
#endif
        // save Trial
        QString trial_filename = inputPath + OS_SEP + "trial.cfg";
        tracker->trial()->save(trial_filename.toStdString().c_str());
        xmlWriter.writeStartElement("Trial");
        xmlWriter.writeCharacters(trial_filename);
        xmlWriter.writeEndElement();

        // save Filters
        filters_widget->saveAllSettings(inputPath + OS_SEP);

        // save Pivot
        for (int i = 0; i < tracker->trial()->num_volumes; i++) {
          xmlWriter.writeStartElement("Pivot");
          xmlWriter.writeAttribute("id", QString::number(i));
          xmlWriter.writeCharacters(QString::fromStdString(tracker->trial()->getVolumeMatrix(i)->to_string()));
          xmlWriter.writeEndElement();
        }

        // save Tracking
        ImportExportTrackingOptionsDialog* diag = new ImportExportTrackingOptionsDialog(this);
        diag->exec();

        bool save_as_matrix = diag->diag->radioButton_TypeMatrix->isChecked();
        bool save_as_rows = diag->diag->radioButton_OrientationRow->isChecked();
        bool save_with_commas = diag->diag->radioButton_SeperatorComma->isChecked();
        bool convert_to_cm = diag->diag->radioButton_TranslationCM->isChecked();
        bool convert_to_rad = diag->diag->radioButton_RotationRadians->isChecked();
        bool interpolate = diag->diag->radioButton_InterpolationSpline->isChecked();

        QString tracking_filename = inputPath + OS_SEP + "track_data.cfg";
        save_tracking_results(tracking_filename,
                              save_as_matrix,
                              save_as_rows,
                              save_with_commas,
                              convert_to_cm,
                              convert_to_rad,
                              interpolate);

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

        // save TrackingOptions
        TrackingOptionsDialog* tracking_dialog_tmp;
        tracking_dialog_tmp = new TrackingOptionsDialog(this);
        tracking_dialog_tmp->setRange(timeline_widget->getPosition_graph()->min_frame,
                                      timeline_widget->getPosition_graph()->max_frame,
                                      tracker->trial()->num_frames - 1);
        tracking_dialog_tmp->inActive = true;
        tracking_dialog_tmp->exec();
        xmlWriter.writeStartElement("TrackingOptions");
        xmlWriter.writeAttribute("Start", QString::number(tracking_dialog_tmp->diag->spinBox_FrameStart->value()));
        xmlWriter.writeAttribute("End", QString::number(tracking_dialog_tmp->diag->spinBox_FrameEnd->value()));
        xmlWriter.writeAttribute("Guess", QString::number(getTracker()->trial()->guess));
        xmlWriter.writeAttribute("Iterations",
                                 QString::number(tracking_dialog_tmp->diag->spinBox_NumberRefinements->value()));
        xmlWriter.writeEndElement();

        xmlWriter.writeEndElement();

        delete tracking_dialog_tmp;
        xmlWriter.writeEndDocument();
        file.close();
      }
    }
  }
}

void AutoscoperMainWindow::runBatch(QString batchfile, bool saveData)
{
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

  if (batchfile.isNull() == false) {

    QString xml_filename = batchfile;
    if (!xml_filename.isNull()) {
      QFile file(xml_filename);
      if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QXmlStreamReader xmlReader(&file);
        // Reading from the file

        while (!xmlReader.atEnd()) {
          if (xmlReader.isStartElement()) {
            QString name = xmlReader.name().toString();
            if (name == "GPUDevice") {
#if defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
              fprintf(stderr, "Load GPUDevice Setting\n");
              xromm::gpu::setUsedPlatform(xmlReader.readElementText().toInt());
              QApplication::processEvents();
#endif
            } else if (name == "Trial") {
              fprintf(stderr, "Load Trial Setting\n");
              openTrial(xmlReader.readElementText());
              end_Frame = tracker->trial()->num_frames;
              QFileInfo info(file);
              filters_widget->loadAllSettings(info.absolutePath() + OS_SEP);
              QApplication::processEvents();
            } else if (name == "Pivot") {
              QXmlStreamAttributes attr = xmlReader.attributes();
              int id = attr.value("id").toString().toInt();
              fprintf(stderr, "Load Pivot %d Setting\n", id);
              QString pivot_data = xmlReader.readElementText();
              tracker->trial()->getVolumeMatrix(id)->from_string(pivot_data.toStdString().c_str());
            } else if (name == "TrackingData") {
              fprintf(stderr, "Load TrackingData Setting\n");
              QXmlStreamAttributes attr = xmlReader.attributes();
              save_as_matrix = attr.value("Matrix").toString().toInt();
              save_as_rows = attr.value("Rows").toString().toInt();
              save_with_commas = attr.value("Commas").toString().toInt();
              convert_to_cm = attr.value("cm").toString().toInt();
              convert_to_rad = attr.value("rad").toString().toInt();
              interpolate = attr.value("interpolate").toString().toInt();
              trackdata_filename = xmlReader.readElementText();

              load_tracking_results(trackdata_filename,
                                    save_as_matrix,
                                    save_as_rows,
                                    save_with_commas,
                                    convert_to_cm,
                                    convert_to_rad,
                                    interpolate);
              QApplication::processEvents();
            } else if (name == "TrackingOptions") {
              fprintf(stderr, "Load TrackingOptions Setting\n");
              doTracking = true;
              QXmlStreamAttributes attr = xmlReader.attributes();
              start_Frame = attr.value("Start").toString().toInt();
              end_Frame = attr.value("End").toString().toInt();
              getTracker()->trial()->guess = attr.value("Guess").toString().toInt();
              iterations = attr.value("Iterations").toString().toInt();
            }
            xmlReader.readNextStartElement();
          } else {
            xmlReader.readNext();
          }
        }
        if (xmlReader.hasError()) {
          std::cout << "XML error: " << xmlReader.error() << std::endl;
        }
        file.close();
      }
    }
  }

  if (doTracking) {
    fprintf(stderr, "Start Tracking\n");
    TrackingOptionsDialog* tracking_dialog_tmp;
    tracking_dialog_tmp = new TrackingOptionsDialog(this);
    tracking_dialog_tmp->diag->spinBox_FrameStart->setValue(start_Frame);
    tracking_dialog_tmp->diag->spinBox_FrameEnd->setValue(end_Frame);
    tracking_dialog_tmp->diag->spinBox_NumberRefinements->setValue(iterations);
    QApplication::processEvents();
    tracking_dialog_tmp->on_pushButton_OK_clicked(true);

    if (saveData) {

      QFileInfo info(trackdata_filename);
      QString tracking_filename_out = info.absolutePath() + OS_SEP + info.completeBaseName() + "_tracked.tra";
      fprintf(stderr, "Save Data to %s\n", tracking_filename_out.toStdString().c_str());
      save_tracking_results(tracking_filename_out,
                            save_as_matrix,
                            save_as_rows,
                            save_with_commas,
                            convert_to_cm,
                            convert_to_rad,
                            interpolate);
    }

    delete tracking_dialog_tmp;
  }
}

void AutoscoperMainWindow::on_actionLoad_xml_batch_triggered(bool checked)
{

  QString inputfile =
    QFileDialog::getOpenFileName(this, tr("Open XML File"), QDir::currentPath(), tr("XML Files (") + tr(" *.xml)"));
  if (inputfile.isNull() == false) {
    fprintf(stderr, "%s\n", inputfile.toStdString().c_str());
    runBatch(inputfile);
  }
}

// Edit menu
void AutoscoperMainWindow::on_actionUndo_triggered(bool checked)
{
  undo_state();
}

void AutoscoperMainWindow::on_actionRedo_triggered(bool checked)
{
  redo_state();
}

void AutoscoperMainWindow::on_actionCut_triggered(bool checked)
{
  push_state();

  if (!timeline_widget->getSelectedNodes()->empty()) {
    timeline_widget->getCopiedNodes()->clear();
    for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
      if ((*timeline_widget->getSelectedNodes())[i].second == NODE) {
        timeline_widget->getCopiedNodes()->push_back((*timeline_widget->getSelectedNodes())[i].first);
        if (!timeline_widget->getPosition_graph()->frame_locks.at(
              (int)(*timeline_widget->getSelectedNodes())[i].first.first->time(
                (*timeline_widget->getSelectedNodes())[i].first.second))) {
          (*timeline_widget->getSelectedNodes())[i].first.first->erase(
            (*timeline_widget->getSelectedNodes())[i].first.second);
        }
      }
    }
    timeline_widget->getSelectedNodes()->clear();
  }

  update_xyzypr_and_coord_frame();
  update_graph_min_max(timeline_widget->getPosition_graph());

  redrawGL();
}

void AutoscoperMainWindow::on_actionCopy_triggered(bool checked)
{
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

void AutoscoperMainWindow::on_actionPaste_triggered(bool checked)
{
  push_state();

  if (!timeline_widget->getCopiedNodes()->empty()) {
    float frame_offset =
      timeline_widget->getCopiedNodes()->front().first->time(timeline_widget->getCopiedNodes()->front().second);
    for (unsigned i = 0; i < timeline_widget->getCopiedNodes()->size(); i++) {
      float frame =
        tracker->trial()->frame
        + (*timeline_widget->getCopiedNodes())[i].first->time((*timeline_widget->getCopiedNodes())[i].second)
        - frame_offset;
      if (!timeline_widget->getPosition_graph()->frame_locks.at((int)frame)) {
        if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve<float>::X_CURVE) {
          KeyCurve<float>* x_curve = dynamic_cast<KeyCurve<float>*>((*timeline_widget->getCopiedNodes())[i].first);
          getTracker()->trial()->getXCurve(-1)->insert(
            frame, (x_curve->value((*timeline_widget->getCopiedNodes())[i].second)));
        } else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve<float>::Y_CURVE) {
          KeyCurve<float>* y_curve = dynamic_cast<KeyCurve<float>*>((*timeline_widget->getCopiedNodes())[i].first);
          getTracker()->trial()->getYCurve(-1)->insert(
            frame, (y_curve->value((*timeline_widget->getCopiedNodes())[i].second)));
        } else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve<float>::Z_CURVE) {
          KeyCurve<float>* z_curve = dynamic_cast<KeyCurve<float>*>((*timeline_widget->getCopiedNodes())[i].first);
          getTracker()->trial()->getZCurve(-1)->insert(
            frame, (z_curve->value((*timeline_widget->getCopiedNodes())[i].second)));
        } else if ((*timeline_widget->getCopiedNodes())[i].first->type == KeyCurve<Quatf>::QUAT_CURVE) {
          KeyCurve<Quatf>* quat_curve = dynamic_cast<KeyCurve<Quatf>*>((*timeline_widget->getCopiedNodes())[i].first);
          getTracker()->trial()->getQuatCurve(-1)->insert(
            frame, (quat_curve->value((*timeline_widget->getCopiedNodes())[i].second)));
        }
      }
    }
    timeline_widget->getSelectedNodes()->clear();
  }

  update_xyzypr_and_coord_frame();
  update_graph_min_max(timeline_widget->getPosition_graph());

  redrawGL();
}

void AutoscoperMainWindow::on_actionDelete_triggered(bool checked)
{
  if (timeline_widget->getSelectedNodes()->empty()) {
    return;
  }
  push_state();

  // std::cout << "Size\n: " << timeline_widget->getSelectedNodes()->size();
  for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
    if ((*timeline_widget->getSelectedNodes())[i].second == NODE) {
      if (!timeline_widget->getPosition_graph()->frame_locks.at(
            (int)(*timeline_widget->getSelectedNodes())[i].first.first->time(
              (*timeline_widget->getSelectedNodes())[i].first.second))) {
        (*timeline_widget->getSelectedNodes())[i].first.first->erase(
          (*timeline_widget->getSelectedNodes())[i].first.second);
      }
    }
  }
  timeline_widget->getSelectedNodes()->clear();

  update_xyzypr_and_coord_frame();
  update_graph_min_max(timeline_widget->getPosition_graph());

  redrawGL();
}

void AutoscoperMainWindow::on_actionSet_Background_triggered(bool checked)
{
  bool ok;
  double threshold = QInputDialog::getDouble(this, "Enter threshold", "threshold in percent", 0.2, 0.0, 1.0, 4, &ok);
  if (ok) {
    setBackground(threshold);
  }
}

// Tracking menu
void AutoscoperMainWindow::on_actionImport_triggered(bool checked)
{
  QString filename = get_filename(false, "*.tra");
  if (filename.compare("") != 0) {
    load_tracking_results(filename);
  }
}

void AutoscoperMainWindow::on_actionExport_triggered(bool checked)
{
  QString filename = get_filename(true, "*.tra");
  if (filename.compare("") != 0) {
    save_tracking_results(filename);
  }
}

void AutoscoperMainWindow::on_actionExport_Full_DRR_Image_triggered(bool checked)
{
  this->saveFullDRR();
}

void AutoscoperMainWindow::on_actionInsert_Key_triggered(bool checked)
{
  push_state();
  timeline_widget->getSelectedNodes()->clear();

  double xyzypr[6];
  double manip_0[6] = { 0 };
  (CoordFrame::from_matrix(trans(getManipulator()->transform())) * *tracker->trial()->getVolumeMatrix(-1))
    .to_xyzypr(xyzypr);
  getTracker()->trial()->getXCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[0]);
  getTracker()->trial()->getYCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[1]);
  getTracker()->trial()->getZCurve(-1)->insert(getTracker()->trial()->frame, xyzypr[2]);
  getTracker()->trial()->getQuatCurve(-1)->insert(getTracker()->trial()->frame, Quatf(xyzypr[3], xyzypr[4], xyzypr[5]));

  timeline_widget->update_graph_min_max();

  redrawGL();

  double NCC = getTracker()->minimizationFunc(manip_0);
  std::cout << "Current NCC is: " << NCC << std::endl;
}

void AutoscoperMainWindow::on_actionLock_triggered(bool checked)
{
  push_state();

  for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {

    int time = (int)(*timeline_widget->getSelectedNodes())[i].first.first->time(
      (*timeline_widget->getSelectedNodes())[i].first.second);

    // Force the addition of keys for all curves in order to truly freeze
    // the frame

    tracker->trial()->getXCurve(-1)->insert(time);
    tracker->trial()->getYCurve(-1)->insert(time);
    tracker->trial()->getZCurve(-1)->insert(time);
    tracker->trial()->getQuatCurve(-1)->insert(time);

    timeline_widget->getPosition_graph()->frame_locks.at(time) = true;
  }

  timeline_widget->getSelectedNodes()->clear();

  frame_changed();
  redrawGL();
}

void AutoscoperMainWindow::on_actionUnlock_triggered(bool checked)
{
  for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
    timeline_widget->getPosition_graph()->frame_locks.at(
      (int)(*timeline_widget->getSelectedNodes())[i].first.first->time(
        (*timeline_widget->getSelectedNodes())[i].first.second)) = false;
  }

  frame_changed();
  redrawGL();
}

void AutoscoperMainWindow::on_actionBreak_Tangents_triggered(bool checked)
{
  push_state();

  for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
    IKeyCurve* curve = (*timeline_widget->getSelectedNodes())[i].first.first;
    IKeyCurve::iterator it = (*timeline_widget->getSelectedNodes())[i].first.second;
    if (!timeline_widget->getPosition_graph()->frame_locks.at((int)curve->time(it))) {
      curve->set_bind_tangents(it, false);
    }
  }
}

void AutoscoperMainWindow::on_actionSmooth_Tangents_triggered(bool checked)
{
  push_state();

  /*for (unsigned i = 0; i < timeline_widget->getSelectedNodes()->size(); i++) {
      KeyCurve& curve = *(*timeline_widget->getSelectedNodes())[i].first.first;
      KeyCurve::iterator it = (*timeline_widget->getSelectedNodes())[i].first.second;

      if (!timeline_widget->getPosition_graph()->frame_locks.at((int)curve.time(it))) {
          curve.set_bind_tangents(it,true);
          curve.set_in_tangent_type(it,KeyCurve::SMOOTH);
          curve.set_out_tangent_type(it,KeyCurve::SMOOTH);
      }
     }*/

  // BARDIYA ADDED MOVING AVERAGE FILTER
  int nWin = 5; // Default
  // puts("change nWin in actionSmooth function");
  MovingAverageFilter(nWin, 0, tracker->trial()->num_frames);

  update_xyzypr_and_coord_frame();
  update_graph_min_max(timeline_widget->getPosition_graph());

  redrawGL();
}

// Advanced Settings
void AutoscoperMainWindow::on_actionAdvanced_Settings_triggered(bool checked)
{
  if (advanced_dialog == NULL)
    advanced_dialog = new AdvancedOptionsDialog(this);

  advanced_dialog->setRangeAdvanced(timeline_widget->getPosition_graph()->min_frame,
                                    timeline_widget->getPosition_graph()->max_frame,
                                    tracker->trial()->num_frames - 1);

  advanced_dialog->setDefPaths(
    default_root_path, default_filter_folder, default_filter_name, default_tracking_folder, default_task_name);

  advanced_dialog->show();
}

// View
void AutoscoperMainWindow::on_actionLayoutCameraViews_triggered(bool triggered)
{

  bool ok;
  int rows = QInputDialog::getInt(this, tr("Layout Camera Views"), tr("Number of Rows"), 1, 1, 10, 1, &ok);
  if (ok)
    relayoutCameras(rows);
}

void AutoscoperMainWindow::on_actionShow_world_view_triggered(bool checked)
{
  if (checked) {
    worldview->show();
  } else {
    worldview->hide();
  }
}

void AutoscoperMainWindow::on_actionThick_Lines_Mode_triggered(bool checked)
{
  // Check if we have at least one Manipluator
  if (manipulator.size() == 0) {
    return;
  }

  // Set Thick Lines Mode for all Manipulators
  for (Manip3D* manip3D : manipulator) {
    manip3D->setThickLinesMode(checked);
  }

  // Redraw
  redrawGL();
}

// Toolbar
void AutoscoperMainWindow::on_toolButtonOpenTrial_clicked()
{
  save_trial_prompt();
  save_tracking_prompt();

  openTrial();
}

void AutoscoperMainWindow::on_toolButtonSaveTracking_clicked()
{
  QString filename = get_filename(true, "*.tra");
  if (filename.compare("") != 0) {
    save_tracking_results(filename);
  }
}

void AutoscoperMainWindow::on_toolButtonLoadTracking_clicked()
{
  QString filename = get_filename(false, "*.tra");
  if (filename.compare("") != 0) {
    load_tracking_results(filename);
  }
}

void AutoscoperMainWindow::on_toolButtonTranslate_clicked()
{
  ui->toolButtonTranslate->setChecked(true);
  ui->toolButtonRotate->setChecked(false);
  getManipulator()->set_mode(Manip3D::TRANSLATION);
  redrawGL();
}

void AutoscoperMainWindow::on_toolButtonRotate_clicked()
{
  ui->toolButtonRotate->setChecked(true);
  ui->toolButtonTranslate->setChecked(false);
  getManipulator()->set_mode(Manip3D::ROTATION);
  redrawGL();
}

void AutoscoperMainWindow::on_toolButtonMovePivot_clicked()
{
  getManipulator()->set_movePivot(ui->toolButtonMovePivot->isChecked());
  redrawGL();
}

void AutoscoperMainWindow::on_toolButtonTrack_clicked()
{
  if (tracking_dialog == NULL)
    tracking_dialog = new TrackingOptionsDialog(this);

  tracking_dialog->setRange(timeline_widget->getPosition_graph()->min_frame,
                            timeline_widget->getPosition_graph()->max_frame,
                            tracker->trial()->num_frames - 1);

  tracking_dialog->show();
}

void AutoscoperMainWindow::on_toolButtonTrackCurrent_clicked()
{
  if (tracking_dialog == NULL)
    tracking_dialog = new TrackingOptionsDialog(this);

  tracking_dialog->setRange(timeline_widget->getPosition_graph()->min_frame,
                            timeline_widget->getPosition_graph()->max_frame,
                            tracker->trial()->num_frames - 1);

  tracking_dialog->trackCurrent();
}

/*void AutoscoperMainWindow::on_toolButtonRetrack_clicked(){
   if(tracking_dialog == NULL)
    tracking_dialog = new TrackingOptionsDialog(this);

   tracking_dialog->setRange(timeline_widget->getPosition_graph()->min_frame,timeline_widget->getPosition_graph()->max_frame,
    tracker->trial()->num_frames-1);

   tracking_dialog->retrack();
   }*/

void AutoscoperMainWindow::on_toolButtonPreviousCurveSet_clicked()
{
  if (tracker) {
    tracker->trial()->setCurrentCurveSetToPrevious();
  }
  redrawGL();
  frame_changed();
}

void AutoscoperMainWindow::on_toolButtonAddNewCurveSet_clicked()
{
  if (tracker)
    tracker->trial()->addCurveSet();
  redrawGL();
  frame_changed();
}

void AutoscoperMainWindow::on_toolButtonNextCurveSet_clicked()
{
  if (tracker) {
    tracker->trial()->setCurrentCurveSetToNext();
  }
  redrawGL();
  frame_changed();
}

void AutoscoperMainWindow::on_actionExport_NCC_as_csv_triggered(bool checked)
{
  QString filename = get_filename(true, "*.ncc");
  if (filename.compare("") != 0) {
    save_ncc_results(filename);
  }
}

void AutoscoperMainWindow::save_ncc_results(QString filename)
{

  std::ofstream file(filename.toStdString().c_str(), std::ios::out);

  file.precision(6);
  file.setf(std::ios::fixed, std::ios::floatfield);
  unsigned int volume = tracker->trial()->current_volume;
  std::vector<double> ncc_values(2, 999);
  for (int frame = 0; frame < tracker->trial()->num_frames; ++frame) {
    float x_val = (*tracker->trial()->getXCurve(volume))(frame);
    float y_val = (*tracker->trial()->getYCurve(volume))(frame);
    float z_val = (*tracker->trial()->getZCurve(volume))(frame);
    Quatf quat_val = (*tracker->trial()->getQuatCurve(volume))(frame);
    Vec3f eulers = quat_val.toEuler();

    double pose[6] = { x_val, y_val, z_val, eulers.z, eulers.y, eulers.x };

    setFrame(frame);
    ncc_values = tracker->trackFrame(volume, pose);

    file << ncc_values.at(0) << "," << ncc_values.at(1) << "," << ncc_values.at(0) + ncc_values.at(1) << std::endl;
  }
  file.close();
}

void AutoscoperMainWindow::on_actionExport_all_NCCs_near_this_pose_triggered(bool checked)
{
  QString filename = get_filename(true, "*.ncc");
  if (filename.compare("") != 0) {
    save_nearby_nccs(filename);
  }
}

void AutoscoperMainWindow::on_actionAboutAutoscoper_triggered(bool checked)
{
  /*puts("Copyright (c) 2011-2019, Brown University\n\
     All rights reserved.\n\
     This is autoscoper 2 by Dr.Ben Knorlein, and modified by Bardiya Akhbari.\n\
     Autoscoper 1 was developed by Andy Loomis(original CUDA version) and Mark Howison(OpenCL reimplementation).");*/

  if (about_autoscoper == NULL) {
    about_autoscoper = new AboutAutoscoper(this);
  }

  about_autoscoper->show();
}

void AutoscoperMainWindow::on_actionOpen_Sample_Wrist_triggered(bool checked)
{

  QString root_path = qApp->applicationDirPath() + "/";
  // "/Users/bardiya/autoscoper-v2";// QDir::currentPath(); //qApp->applicationDirPath();

  QString default_config_path = root_path + "sample_data";
  default_config_path += "/";
  default_config_path += "wrist.cfg";

  std::ifstream file(default_config_path.toStdString().c_str());
  if (file.is_open() == false) {
    QString l_1 = "mayaCam_csv " + root_path + "sample_data/Calibration/xr_calib_wrist_cam01.txt";
    QString l_2 = "mayaCam_csv " + root_path + "sample_data/Calibration/xr_calib_wrist_cam02.txt";
    QString l_3 = "CameraRootDir " + root_path + "sample_data/XMA_UND/xr_data_wrist_cam01";
    QString l_4 = "CameraRootDir " + root_path + "sample_data/XMA_UND/xr_data_wrist_cam02";
    QString l_5 = "VolumeFile " + root_path + "sample_data/Models/rad_dcm_cropped.tif";
    QString l_6 = "VolumeFlip 0 0 0";
    QString l_7 = "VoxelSize 0.39625 0.39625 0.625";
    QString l_8 = "RenderResolution 512 512";
    QString l_9 = "OptimizationOffsets 0.1 0.1 0.1 0.1 0.1 0.1";

    QString l_10 = "VolumeFile " + root_path + "sample_data/Models/mc2_mc3_dcm_cropped.tif";
    QString l_11 = "VolumeFile " + root_path + "sample_data/Models/mc3_dcm_cropped.tif";
    QString l_12 = "VolumeFile " + root_path + "sample_data/Models/uln_dcm_cropped.tif";

    std::ofstream cfg_file(default_config_path.toStdString().c_str());
    cfg_file.precision(12);
    cfg_file << l_1.toStdString().c_str() << std::endl;
    cfg_file << l_2.toStdString().c_str() << std::endl;
    cfg_file << l_3.toStdString().c_str() << std::endl;
    cfg_file << l_4.toStdString().c_str() << std::endl;
    // Radius
    cfg_file << l_5.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;
    // MC2-MC3
    cfg_file << l_10.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;
    // MC3
    cfg_file << l_11.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;
    // ULN
    cfg_file << l_12.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;
    cfg_file << l_8.toStdString().c_str() << std::endl;
    cfg_file << l_9.toStdString().c_str();
    cfg_file.close();
  }
  file.close();

  openTrial(default_config_path);
}

void AutoscoperMainWindow::on_actionOpen_Sample_Knee_triggered(bool checked)
{

  QString root_path = qApp->applicationDirPath() + "/";

  QString default_config_path = root_path + "sample_data";
  default_config_path += "/";
  default_config_path += "left_knee.cfg";

  std::ifstream file(default_config_path.toStdString().c_str());
  if (file.is_open() == false) {
    QString l_1 = "mayaCam_csv " + root_path + "sample_data/Calibration/xr_calib_left_knee_cam01.txt";
    QString l_2 = "mayaCam_csv " + root_path + "sample_data/Calibration/xr_calib_left_knee_cam02.txt";
    QString l_3 = "CameraRootDir " + root_path + "sample_data/XMA_UND/xr_data_left_knee_cam01";
    QString l_4 = "CameraRootDir " + root_path + "sample_data/XMA_UND/xr_data_left_knee_cam02";
    QString l_5 = "VolumeFile " + root_path + "sample_data/Models/left_knee_femur_cropped.tif";
    QString l_6 = "VolumeFlip 0 0 0";
    QString l_7 = "VoxelSize 0.421875 0.421875 0.625";
    QString l_8 = "RenderResolution 512 512";
    QString l_9 = "OptimizationOffsets 0.1 0.1 0.1 0.1 0.1 0.1";

    QString l_10 = "VolumeFile " + root_path + "sample_data/Models/left_knee_tibia_cropped.tif";

    std::ofstream cfg_file(default_config_path.toStdString().c_str());
    cfg_file.precision(12);
    cfg_file << l_1.toStdString().c_str() << std::endl;
    cfg_file << l_2.toStdString().c_str() << std::endl;
    cfg_file << l_3.toStdString().c_str() << std::endl;
    cfg_file << l_4.toStdString().c_str() << std::endl;
    // Femur
    cfg_file << l_5.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;
    // Tibia
    cfg_file << l_10.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;

    cfg_file << l_8.toStdString().c_str() << std::endl;
    cfg_file << l_9.toStdString().c_str();
    cfg_file.close();
  }
  file.close();

  // cout << default_config_path.toStdString().c_str() << std::endl;
  openTrial(default_config_path);
}

void AutoscoperMainWindow::on_actionOpen_Sample_Ankle_triggered(bool checked)
{

  QString root_path = qApp->applicationDirPath() + "/";

  QString default_config_path = root_path + "sample_data";
  default_config_path += "/";
  default_config_path += "right_ankle.cfg";

  std::ifstream file(default_config_path.toStdString().c_str());
  if (file.is_open() == false) {
    QString l_1 = "mayaCam_csv " + root_path + "sample_data/Calibration/xr_calib_right_ankle_cam01.txt";
    QString l_2 = "mayaCam_csv " + root_path + "sample_data/Calibration/xr_calib_right_ankle_cam02.txt";
    QString l_3 = "CameraRootDir " + root_path + "sample_data/XMA_UND/xr_data_right_ankle_cam01";
    QString l_4 = "CameraRootDir " + root_path + "sample_data/XMA_UND/xr_data_right_ankle_cam02";
    QString l_5 = "VolumeFile " + root_path + "sample_data/Models/right_ankle_calc.tif";
    QString l_6 = "VolumeFlip 0 0 0";
    QString l_7 = "VoxelSize 0.4414 0.4414 0.625";
    QString l_8 = "RenderResolution 512 512";
    QString l_9 = "OptimizationOffsets 0.1 0.1 0.1 0.1 0.1 0.1";

    QString l_10 = "VolumeFile " + root_path + "sample_data/Models/right_ankle_talus.tif";
    QString l_11 = "VolumeFile " + root_path + "sample_data/Models/right_ankle_tibia.tif";

    std::ofstream cfg_file(default_config_path.toStdString().c_str());
    cfg_file.precision(12);
    cfg_file << l_1.toStdString().c_str() << std::endl;
    cfg_file << l_2.toStdString().c_str() << std::endl;
    cfg_file << l_3.toStdString().c_str() << std::endl;
    cfg_file << l_4.toStdString().c_str() << std::endl;
    // Calc
    cfg_file << l_5.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;
    // Talus
    cfg_file << l_10.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;
    // Tibia
    cfg_file << l_11.toStdString().c_str() << std::endl;
    cfg_file << l_6.toStdString().c_str() << std::endl;
    cfg_file << l_7.toStdString().c_str() << std::endl;

    cfg_file << l_8.toStdString().c_str() << std::endl;
    cfg_file << l_9.toStdString().c_str();
    cfg_file.close();
  }
  file.close();

  // cout << default_config_path.toStdString().c_str() << std::endl;
  openTrial(default_config_path);
}

void AutoscoperMainWindow::save_nearby_nccs(QString filename)
{
  std::ofstream file(filename.toStdString().c_str(), std::ios::out);

  file.precision(6);
  file.setf(std::ios::fixed, std::ios::floatfield);
  unsigned int volume = tracker->trial()->current_volume;
  std::vector<double> ncc_values(2, 9999);

  float x_val = (*tracker->trial()->getXCurve(-1))(curFrame);
  float y_val = (*tracker->trial()->getYCurve(-1))(curFrame);
  float z_val = (*tracker->trial()->getZCurve(-1))(curFrame);
  Quatf quat_val = (*tracker->trial()->getQuatCurve(-1))(curFrame);
  Vec3f eulers = quat_val.toEuler();

  double pose[6] = { x_val, y_val, z_val, eulers.z, eulers.y, eulers.x };

  // int iter_max = 5000;
  // int t_lim = 1;

  // Get Current One
  ncc_values = tracker->trackFrame(volume, pose);
  file << pose[0] << "," << pose[1] << "," << pose[2] << "," << pose[3] << "," << pose[4] << "," << pose[5] << ","
       << ncc_values.at(0) << "," << ncc_values.at(1) << "," << ncc_values.at(0) + ncc_values.at(1) << std::endl;

  // Look the neighbors
  double next_pose[6];
  double t_lim = 3;
  double t_skip = 0.1;
  double r_lim = .3;
  double r_skip = 0.1;

  for (double d1 = -t_lim; d1 <= t_lim; d1 = d1 + t_skip) {
    for (double d2 = -t_lim; d2 <= t_lim; d2 = d2 + t_skip) {
      for (double d3 = -t_lim; d3 <= t_lim; d3 = d3 + t_skip) {
        for (double d4 = -r_lim; d4 <= r_lim; d4 = d4 + r_skip) {
          for (double d5 = -r_lim; d5 <= r_lim; d5 = d5 + r_skip) {
            for (double d6 = -r_lim; d6 <= r_lim; d6 = d6 + r_skip) {
              next_pose[0] = pose[0] + d1;
              next_pose[1] = pose[1] + d2;
              next_pose[2] = pose[2] + d3;
              next_pose[3] = pose[3] + d4;
              next_pose[4] = pose[4] + d5;
              next_pose[5] = pose[5] + d6;

              ncc_values = tracker->trackFrame(volume, next_pose);

              file << next_pose[0] << "," << next_pose[1] << "," << next_pose[2] << "," << next_pose[3] << ","
                   << next_pose[4] << "," << next_pose[5] << "," << ncc_values.at(0) << "," << ncc_values.at(1) << ","
                   << ncc_values.at(0) + ncc_values.at(1) << std::endl;
            }
          }
        }
      }
    }
  }

  file.close();

  std::cout << "All NCCs were saved nearby this pose..." << std::endl;
}

void AutoscoperMainWindow::key_w_pressed()
{
  ui->toolButtonTranslate->click();
}
void AutoscoperMainWindow::key_e_pressed()
{
  ui->toolButtonRotate->click();
}
void AutoscoperMainWindow::key_d_pressed()
{
  ui->toolButtonMovePivot->click();
}
void AutoscoperMainWindow::key_h_pressed()
{
  filters_widget->toggle_drrs();
}
void AutoscoperMainWindow::key_l_pressed()
{
  filters_widget->toggle_radiographs();
}
void AutoscoperMainWindow::key_t_pressed()
{
  ui->toolButtonTrack->click();
}
void AutoscoperMainWindow::key_j_pressed()
{
  ui->toolButtonPreviousCurveSet->click();
}
void AutoscoperMainWindow::key_k_pressed()
{
  ui->toolButtonNextCurveSet->click();
}
/*void AutoscoperMainWindow::key_p_pressed(){
   ui->toolButtonRetrack->click();
   } */
void AutoscoperMainWindow::key_c_pressed()
{
  on_actionInsert_Key_triggered(true); // Insert the current frame and then run the tracking
  ui->toolButtonTrackCurrent->click();
}
void AutoscoperMainWindow::key_plus_pressed()
{
  getManipulator(-1)->set_pivotSize(getManipulator(-1)->get_pivotSize() * 1.1f);
  redrawGL();
}
void AutoscoperMainWindow::key_equal_pressed()
{
  getManipulator(-1)->set_pivotSize(getManipulator(-1)->get_pivotSize() * 1.1f);
  redrawGL();
}

void AutoscoperMainWindow::key_minus_pressed()
{
  getManipulator(-1)->set_pivotSize(getManipulator(-1)->get_pivotSize() * 0.9f);
  redrawGL();
}

void AutoscoperMainWindow::key_left_pressed()
{
  // Move to the prev DOF
  int updatedDOF = this->timeline_widget->getCurrentDOF();
  updatedDOF--;
  if (updatedDOF < 0) {
    updatedDOF = TimelineDockWidget::NUMBER_OF_DEGREES_OF_FREEDOM - 1;
  }
  this->timeline_widget->setCurrentDOF(updatedDOF);
}

void AutoscoperMainWindow::key_right_pressed()
{
  // Move to the next DOF
  int updatedDOF = timeline_widget->getCurrentDOF();
  updatedDOF++;
  if (updatedDOF >= TimelineDockWidget::NUMBER_OF_DEGREES_OF_FREEDOM) {
    updatedDOF = 0;
  }
  timeline_widget->setCurrentDOF(updatedDOF);
}

void AutoscoperMainWindow::key_up_pressed()
{
  std::vector<double> curPose = getPose(-1, -1);
  int currentDOF = this->timeline_widget->getCurrentDOF();
  curPose[currentDOF] = curPose[currentDOF] + stepVal;
  setPose(curPose, -1, -1);
}

void AutoscoperMainWindow::key_down_pressed()
{
  std::vector<double> curPose = getPose(-1, -1);
  int currentDOF = this->timeline_widget->getCurrentDOF();
  curPose[currentDOF] = curPose[currentDOF] - stepVal;
  setPose(curPose, -1, -1);
}

// Shortcuts
void AutoscoperMainWindow::setupShortcuts()
{
  // W - switches to translate mode
  new QShortcut(QKeySequence(Qt::Key_W), this, SLOT(key_w_pressed()));
  // E - switches to rotate mode
  new QShortcut(QKeySequence(Qt::Key_E), this, SLOT(key_e_pressed()));
  // D - switches to move pivot mode
  new QShortcut(QKeySequence(Qt::Key_D), this, SLOT(key_d_pressed()));
  // H - Enables/Disables DRRs
  new QShortcut(QKeySequence(Qt::Key_H), this, SLOT(key_h_pressed()));
  // L - Enable/Disable Radiographs
  new QShortcut(QKeySequence(Qt::Key_L), this, SLOT(key_l_pressed()));
  // T - Opens the tracking dialog
  new QShortcut(QKeySequence(Qt::Key_T), this, SLOT(key_t_pressed()));
  // J - Goes to the previous tracking set if there is one
  new QShortcut(QKeySequence(Qt::Key_J), this, SLOT(key_j_pressed()));
  // K - Goes to the next tracking set if there is one
  new QShortcut(QKeySequence(Qt::Key_K), this, SLOT(key_k_pressed()));
  // P - Retrack - not enabled
  // new QShortcut(QKeySequence(Qt::Key_P), this, SLOT(key_p_pressed()));
  // C - Track current frame
  new QShortcut(QKeySequence(Qt::Key_C), this, SLOT(key_c_pressed()));
  // + - Increase pivot size
  new QShortcut(QKeySequence(Qt::Key_Plus), this, SLOT(key_plus_pressed()));
  // = - Increase pivot size
  new QShortcut(QKeySequence(Qt::Key_Equal), this, SLOT(key_equal_pressed()));
  // - - Decrease pivot size
  new QShortcut(QKeySequence(Qt::Key_Minus), this, SLOT(key_minus_pressed()));

  // Left Arrow Key - Previous DOF
  new QShortcut(QKeySequence(Qt::Key_Left), this, SLOT(key_left_pressed()));
  // Right Arrow Key - Next DOF
  new QShortcut(QKeySequence(Qt::Key_Right), this, SLOT(key_right_pressed()));
  // Up Arrow Key - Increment DOF
  new QShortcut(QKeySequence(Qt::Key_Up), this, SLOT(key_up_pressed()));
  // Down Arrow Key - Decrement DOF
  new QShortcut(QKeySequence(Qt::Key_Down), this, SLOT(key_down_pressed()));

  // CTRL+C - Copy keyframe
  ui->actionCopy->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_C));
  // CTRL+V - Paste keyframe
  ui->actionPaste->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_V));
  // CTRL+X - Cut keyframe
  ui->actionCut->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_X));
  // Delete - Deletes keyframe
  ui->actionDelete->setShortcut(QKeySequence(Qt::Key_Delete));
  // CTRL+Z - Undo back to last keyframe
  ui->actionUndo->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Z));
  // CTRL+Y - Redo back to next keyframe
  ui->actionRedo->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Y));
  // CTRL+L - Smooth tangents
  ui->actionSmooth_Tangents->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_L));
  // CTRL+N - New trial
  ui->actionNew->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_N));
  // CTRL+O - Open trial
  ui->actionOpen->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));
  // CTRL+S - Save tracking
  ui->actionSave->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_S));
  // CTRL+SHIFT+S - Save trial as
  ui->actionSave_as->setShortcut(QKeySequence(Qt::SHIFT + Qt::CTRL + Qt::Key_S));
  // CTRL+Q - Quit
  ui->actionQuit->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q));
  // S - Insert keyframe
  ui->actionInsert_Key->setShortcut(QKeySequence(Qt::Key_S));
}

int AutoscoperMainWindow::getNumVolumes()
{
  return tracker->trial()->num_volumes;
}

int AutoscoperMainWindow::getNumFrames()
{
  return tracker->trial()->num_frames;
}

void AutoscoperMainWindow::setVolumeVisibility(int volume, bool visible)
{
  volumes_widget->setVolumeVisibility(volume, visible);
}
