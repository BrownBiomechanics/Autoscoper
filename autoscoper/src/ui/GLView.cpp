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
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY �AS IS� WITH NO
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

/// \file GLView.cpp
/// \author Benjamin Knorlein, Andy Loomis

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#ifdef _WIN32
  #include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "ui/GLView.h"
#include "ui/AutoscoperMainWindow.h"
#include "ui/CameraViewWidget.h"
#include "ui/WorldViewWindow.h"
#include "ui/TimelineDockWidget.h"

#include "Tracker.hpp"
#include "View.hpp"
#include "Trial.hpp"

#include "Manip3D.hpp"


#ifdef WITH_CUDA
#include <gpu/cuda/RadRenderer.hpp>
#include <gpu/cuda/RayCaster.hpp>
#else
#include <gpu/opencl/RadRenderer.hpp>
#include <gpu/opencl/RayCaster.hpp>
#endif

#include <QMouseEvent>
#include <QWheelEvent>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

GLView::GLView(QWidget *parent)
    : GLWidget(parent)
{
  m_view = NULL;
  viewdata.m_isStaticView = false;
  double xyzypr[6] = {250.0f, 250.0f, 250.0f, 0.0f, 45.0f, -35.0f};
  defaultViewMatrix = CoordFrame::from_xyzypr(xyzypr);
}

void GLView::setStaticView(bool staticView){
  viewdata.m_isStaticView = staticView;
  if(viewdata.m_isStaticView){
    const double xyzypr[6] = {250.0f, 250.0f, 250.0f, 0.0f, 45.0f, -35.0f};
    defaultViewMatrix = CoordFrame::from_xyzypr(xyzypr);

    viewdata.ratio = 1.0f;
    viewdata.fovy = 53.13f;
    viewdata.near_clip = 1.0f;
    viewdata.far_clip = 10000.0f;
  }
}

void GLView::setView(View * view){
  m_view = view;

  viewdata.ratio = 1.0f;
    viewdata.fovy = 53.13f;
    viewdata.near_clip = 1.0f;
    viewdata.far_clip = 10000.0f;
}

// Selects the axis of translation or rotation of the manipulator that is under
// the mouse located at pixel coordinates x,y.
void GLView::select_manip_in_view(double x, double y, int button)
{
  AutoscoperMainWindow * mainwindow;
  CameraViewWidget * cameraViewWidget;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }


    // Setup the view from this perspective so that we can simply call set_view
    // on the manipulator
    glViewport(viewdata.viewport_x,
               viewdata.viewport_y,
               viewdata.viewport_width,
               viewdata.viewport_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(viewdata.fovy,viewdata.ratio,viewdata.near_clip,viewdata.far_clip);

  CoordFrame viewMatrix;
  if (viewdata.m_isStaticView) {
        viewMatrix = defaultViewMatrix;
  }else{
    viewMatrix = mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame();
  }
    double m[16];
    viewMatrix.inverse().to_matrix(m);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(m);

  if (mainwindow->getManipulator()){
    mainwindow->getManipulator()->set_view();
    mainwindow->getManipulator()->set_size(viewdata.scale*mainwindow->getManipulator()->get_pivotSize());
    mainwindow->getManipulator()->on_mouse_press(x, viewdata.window_height - y);
  }
}

 void GLView::wheelEvent(QWheelEvent *e)
 {
  AutoscoperMainWindow * mainwindow;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

   if ( Qt::ControlModifier & e->modifiers() ) {
     if (e->delta() > 0) {
            viewdata.zoom *= 1.1f;
        }
        else if (e->delta() < 0) {
            viewdata.zoom /= 1.1f;
        }

        update_viewport(&viewdata);
        mainwindow->redrawGL();
    }
 }

void GLView::mousePressEvent(QMouseEvent *e){
  AutoscoperMainWindow * mainwindow;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

  press_x = e->x();
    press_y = e->y();
  prevx = e->x();
  prevy = e->y();

    select_manip_in_view(e->x(),e->y(),e->button());

  mainwindow->redrawGL();
}

// Moves the manipulator and volume based on the view, the selected axis, and
// the direction of the motion.
void GLView::move_manip_in_view(double x, double y, bool out_of_plane)
{
  AutoscoperMainWindow * mainwindow;
  CameraViewWidget * cameraViewWidget;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

  if (mainwindow->getPosition_graph()->frame_locks.at(mainwindow->getTracker()->trial()->frame)) {
        return;
    }

  if (mainwindow->getManipulator()) {
    CoordFrame frame;
    if (mainwindow->getManipulator()->get_movePivot()) {
      frame = (CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()))* *mainwindow->getTracker()->trial()->getVolumeMatrix(-1));
    }

    if (!out_of_plane) {
      mainwindow->getManipulator()->set_size(viewdata.scale*mainwindow->getManipulator()->get_pivotSize());
      mainwindow->getManipulator()->on_mouse_move(x, viewdata.window_height - y);
    }
    else if (mainwindow->getManipulator()->selection() == Manip3D::VIEW_PLANE) {
      CoordFrame mmat = CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()));

      CoordFrame viewMatrix;
      if (viewdata.m_isStaticView) {
        viewMatrix = defaultViewMatrix;
      }
      else{
        viewMatrix = mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame();
      }


      double zdir[3] = { mmat.translation()[0] - viewMatrix.translation()[0],
        mmat.translation()[1] - viewMatrix.translation()[1],
        mmat.translation()[2] - viewMatrix.translation()[2] };
      double mag = sqrt(zdir[0] * zdir[0] + zdir[1] * zdir[1] + zdir[2] * zdir[2]);
      zdir[0] /= mag;
      zdir[1] /= mag;
      zdir[2] /= mag;

      double ztrans[3] = { (x - y) / 2.0*zdir[0], (x - y) / 2.0*zdir[1], (x - y) / 2.0*zdir[2] };

      mmat.translate(ztrans);

      double m[16];
      mmat.to_matrix_row_order(m);
      mainwindow->getManipulator()->set_transform(Mat4d(m));

      mainwindow->getManipulator()->set_selection(Manip3D::VIEW_PLANE);
    }

    if (mainwindow->getManipulator()->get_movePivot()) {
      CoordFrame new_manip_matrix = CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()));
      *(mainwindow->getTracker()->trial()->getVolumeMatrix(-1)) = new_manip_matrix.inverse()*frame;
    }
  }
}

void GLView::mouseMoveEvent(QMouseEvent *e){
  AutoscoperMainWindow * mainwindow;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

  double dx = e->x() - prevx;
    double dy = e->y() - prevy;

  double x = e->x();
  double y = e->y();
  if ( Qt::ControlModifier & e->modifiers() ) {
    if (viewdata.m_isStaticView) {
            if (e->buttons() &  Qt::LeftButton) {
                CoordFrame rotationMatrix;
                rotationMatrix.rotate(defaultViewMatrix.rotation()+3,
                                             -dx/2.0);
                rotationMatrix.rotate(defaultViewMatrix.rotation()+0,
                                             -dy/2.0);

                defaultViewMatrix = rotationMatrix*defaultViewMatrix;
            }
            else if (e->buttons() &  Qt::MiddleButton) {
                double xtrans[3] = {-dx*defaultViewMatrix.rotation()[0],
                                    -dx*defaultViewMatrix.rotation()[1],
                                    -dx*defaultViewMatrix.rotation()[2]};
                double ytrans[3] = {dy*defaultViewMatrix.rotation()[3],
                                    dy*defaultViewMatrix.rotation()[4],
                                    dy*defaultViewMatrix.rotation()[5]};

                defaultViewMatrix.translate(xtrans);
                defaultViewMatrix.translate(ytrans);
            }
      else if (e->buttons() &  Qt::RightButton) {
                double ztrans[3] =
                    { (dx-dy)/2.0*defaultViewMatrix.rotation()[6],
                      (dx-dy)/2.0*defaultViewMatrix.rotation()[7],
                      (dx-dy)/2.0*defaultViewMatrix.rotation()[8] };

                defaultViewMatrix.translate(ztrans);
            }
        }
        else {
            if (e->buttons() &  Qt::LeftButton) {
        viewdata.zoom_x -= dx/200/viewdata.zoom;
        viewdata.zoom_y += dy/200/viewdata.zoom;

        update_viewport(&viewdata);
      }
    }
        update_scale_in_view(&viewdata);
    }
    else {
        if (Qt::ShiftModifier & e->modifiers()) {
            if (e->buttons() & Qt::LeftButton) {
                // Only display in one direction
        if (abs(e->x()-press_x) > abs(e->y()-press_y)) {
                    y = press_y;
                }
                else {
                   x = press_x;
                }
                move_manip_in_view(x,y);
            }
        }
        else {
            if (e->buttons() &  Qt::LeftButton) {
                move_manip_in_view(x,y);
            }
      else if (e->buttons() & Qt::RightButton) {
                move_manip_in_view(dx,dy,true);
            }
        }
    }

    mainwindow->update_xyzypr();

    prevx = x;
    prevy = y;
}

void GLView::mouseReleaseEvent(QMouseEvent *e){
  AutoscoperMainWindow * mainwindow;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

  if (mainwindow->getManipulator())
    mainwindow->getManipulator()->on_mouse_release(e->x(),e->y());

    mainwindow->update_graph_min_max(mainwindow->getTracker()->trial()->frame);

  mainwindow->redrawGL();
}

void GLView::paintGL()
{
  AutoscoperMainWindow * mainwindow;
  CameraViewWidget * cameraViewWidget;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

  if(mainwindow){
    update_scale_in_view(&viewdata);
    update_viewport(&viewdata);

    glViewport(viewdata.viewport_x,
           viewdata.viewport_y,
           viewdata.viewport_width,
           viewdata.viewport_height);

    double m[16];

    if(viewdata.m_isStaticView){
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      defaultViewMatrix.inverse().to_matrix(m);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(viewdata.fovy,viewdata.ratio,viewdata.near_clip,viewdata.far_clip);
      //fprintf(stderr, "%lf %lf %lf %lf \n", viewdata.fovy,viewdata.ratio,viewdata.near_clip,viewdata.far_clip);
      glMatrixMode(GL_MODELVIEW);
      glLoadMatrixd(m);


      // Draw background
      float top_color[3] = {0.20f,0.35f,0.50f};
      float bot_color[3] = {0.10f,0.17f,0.25f};

      draw_gradient(top_color,bot_color);
      for (unsigned int i = 0; i < mainwindow->getTracker()->trial()->cameras.size(); ++i) {
        draw_textured_quad(mainwindow->getTracker()->trial()->cameras.at(i).image_plane(),
          (*mainwindow->getTextures())[i]);
      }

      // Draw cameras
      enable_headlight();
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
      for (unsigned int i = 0; i < mainwindow->getTracker()->trial()->cameras.size(); ++i) {

        glPushMatrix();

        double m1[16];
        mainwindow->getTracker()->trial()->cameras.at(i).coord_frame().to_matrix(m1);
        glMultMatrixd(m1);

        float scale = 0.05*sqrt(m1[12]*m1[12]+m1[13]*m1[13]+m1[14]*m1[14]);
        glScalef(scale,scale,scale);

        glColor3f(0.5f, 0.5f, 0.5f);
        draw_camera();

        glPopMatrix();
      }

      draw_manip_from_view(&viewdata);
      glDisable(GL_LIGHTING);

      // Draw grid
      bool drawGrid = true;
      if (drawGrid == true) {
        draw_xz_grid(24, 24, 10.0f);
      }

      if (!mainwindow->getTracker()->views().empty()) {
        float width = 2.0f / viewdata.zoom, height = 2.0f / viewdata.zoom;
        float x = viewdata.zoom_x - width / 2.0f, y = viewdata.zoom_y - height / 2.0f;

        for (int idx_volume = 0; idx_volume < mainwindow->getTracker()->trial()->num_volumes; idx_volume++){
          CoordFrame modelview = defaultViewMatrix.inverse()*CoordFrame::from_matrix(trans(mainwindow->getManipulator(idx_volume)->transform()))* *mainwindow->getTracker()->trial()->getVolumeMatrix(idx_volume);

          double imv[16];
          modelview.inverse().to_matrix_row_order(imv);
          mainwindow->getTracker()->view(0)->drrRenderer(idx_volume)->setInvModelView(imv);

          mainwindow->getTracker()->view(0)->drrRenderer(idx_volume)->setViewport(
            viewdata.ratio*x, y, viewdata.ratio*width, height);
        }

        mainwindow->getTracker()->view(0)->renderDrr(viewdata.pbo,viewdata.window_width,viewdata.window_height);

        glViewport(0, 0, viewdata.window_width, viewdata.window_height);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE,GL_ONE);

        CALL_GL(glRasterPos2i(0, 0));

  #ifdef WITH_CUDA
        CALL_GL(glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, viewdata.pbo));
        CALL_GL(glDrawPixels(viewdata.window_width,
               viewdata.window_height,
               GL_RGB, GL_FLOAT, 0));
        CALL_GL(glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
  #else
        CALL_GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewdata.pbo));
        CALL_GL(glDrawPixels(viewdata.window_width,
               viewdata.window_height,
               GL_RGB, GL_FLOAT, 0));
        CALL_GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
  #endif
        CALL_GL(glDisable(GL_BLEND));
        CALL_GL(glEnable(GL_DEPTH_TEST));
      }
      return;
    }
    else if(cameraViewWidget){
      float temp = 2.0f*sqrt(5.0)*sin(M_PI*viewdata.fovy/360.0);
      float width = temp/viewdata.zoom, height = temp/viewdata.zoom;
      float x = viewdata.zoom_x-width/2.0f, y = viewdata.zoom_y-height/2.0f;

      for (int idx_volume = 0; idx_volume < mainwindow->getTracker()->trial()->num_volumes; idx_volume++){
        CoordFrame modelview = mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().inverse()
          * CoordFrame::from_matrix(trans(mainwindow->getManipulator(idx_volume)->transform()))
          * (*mainwindow->getTracker()->trial()->getVolumeMatrix(idx_volume));
        double imv[16];
        modelview.inverse().to_matrix_row_order(imv);
        int idx = mainwindow->getTracker()->trial()->current_volume;
        m_view->drrRenderer(idx_volume)->setInvModelView(imv);

        m_view->drrRenderer(idx_volume)->setViewport(
          viewdata.ratio*x, y, viewdata.ratio*width, height);
      }

      m_view->radRenderer()->set_viewport(
        viewdata.ratio*x, y, viewdata.ratio*width, height);

      m_view->backgroundRenderer()->set_viewport(
        viewdata.ratio*x, y, viewdata.ratio*width, height);

      m_view->render(viewdata.pbo,viewdata.window_width, viewdata.window_height);

      glViewport(0, 0,viewdata.window_width, viewdata.window_height);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDisable(GL_DEPTH_TEST);
      glRasterPos2i(0, 0);

      #ifdef WITH_CUDA
      CALL_GL(glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, viewdata.pbo));
      CALL_GL(glDrawPixels(viewdata.window_width,
              viewdata.window_height,
              GL_RGB, GL_FLOAT, 0));
      CALL_GL(glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
      #else
      CALL_GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewdata.pbo));
      CALL_GL(glDrawPixels(viewdata.window_width,
              viewdata.window_height,
              GL_RGB, GL_FLOAT, 0));
      CALL_GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
      #endif

      glEnable(GL_DEPTH_TEST);

      glViewport(viewdata.viewport_x,
            viewdata.viewport_y,
            viewdata.viewport_width,
            viewdata.viewport_height);

      mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).
        coord_frame().inverse().to_matrix(m);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(viewdata.fovy,viewdata.ratio,viewdata.near_clip,viewdata.far_clip);
      glMatrixMode(GL_MODELVIEW);
      glLoadMatrixd(m);

      enable_headlight();
      draw_manip_from_view(&viewdata);
      glDisable(GL_LIGHTING);
    }
  }
}

void GLView::saveView(std::string filename){
  CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> (this->parent());
  AutoscoperMainWindow * mainwindow = cameraViewWidget->getMainWindow();

  // Calculate the minimum and maximum values of the bounding box
  // corners after they have been projected onto the view plane
  double min_max[4] = { 1.0, 1.0, -1.0, -1.0 };

  for (int j = 0; j < 4; j++) {
    // Calculate the location of the corner in camera space
    double corner[3];
    m_view->camera()->coord_frame().inverse().point_to_world_space(&m_view->camera()->image_plane()[3 * j], corner);

    // Calculate its projection onto the film plane, where z = -2
    double film_plane[3];
    film_plane[0] = -2 * corner[0] / corner[2];
    film_plane[1] = -2 * corner[1] / corner[2];

    // Update the min and max values
    if (min_max[0] > film_plane[0]) {
      min_max[0] = film_plane[0];
    }
    if (min_max[1] > film_plane[1]) {
      min_max[1] = film_plane[1];
    }
    if (min_max[2] < film_plane[0]) {
      min_max[2] = film_plane[0];
    }
    if (min_max[3] < film_plane[1]) {
      min_max[3] = film_plane[1];
    }
  }

  double viewport[4];
  viewport[0] = min_max[0];
  viewport[1] = min_max[1];
  viewport[2] = min_max[2] - min_max[0];
  viewport[3] = min_max[3] - min_max[1];

  for (int idx_volume = 0; idx_volume < mainwindow->getTracker()->trial()->num_volumes; idx_volume++){
    CoordFrame modelview = mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().inverse()
      * CoordFrame::from_matrix(trans(mainwindow->getManipulator(idx_volume)->transform()))
      * (*mainwindow->getTracker()->trial()->getVolumeMatrix(idx_volume));
    double imv[16];
    modelview.inverse().to_matrix_row_order(imv);
    int idx = mainwindow->getTracker()->trial()->current_volume;
    m_view->drrRenderer(idx_volume)->setInvModelView(imv);

    m_view->drrRenderer(idx_volume)->setViewport(viewport[0], viewport[1],
      viewport[2], viewport[3]);
  }

  m_view->saveImage(filename, m_view->camera()->size()[0], m_view->camera()->size()[1]);
}


void GLView::update_scale_in_view(ViewData* view)
{
    // Determine the distance from the center of the pivot point to the
    // center of the view.
  AutoscoperMainWindow* mainwindow;
  CameraViewWidget * cameraViewWidget;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

  if (mainwindow->getManipulator()){
    CoordFrame mat = CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()));

    double dist_vec[3];
    if (view->m_isStaticView) {
      dist_vec[0] = mat.translation()[0] -
        defaultViewMatrix.translation()[0];
      dist_vec[2] = mat.translation()[1] -
        defaultViewMatrix.translation()[1];
      dist_vec[1] = mat.translation()[2] -
        defaultViewMatrix.translation()[2];
    }
    else {
      dist_vec[0] = mat.translation()[0] -
        mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().translation()[0];
      dist_vec[1] = mat.translation()[1] -
        mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().translation()[1];
      dist_vec[2] = mat.translation()[2] -
        mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().translation()[2];
    }
    double dist = sqrt(dist_vec[0] * dist_vec[0] +
      dist_vec[1] * dist_vec[1] +
      dist_vec[2] * dist_vec[2]);

    // Adjust the size of the pivot based on the distance.
    view->scale = 2.0*dist*tan(view->fovy*M_PI / 360.0)*view->near_clip / view->zoom;
  }
}

void GLView::draw_manip_from_view(const ViewData* view)
{
  AutoscoperMainWindow * mainwindow;
  if (viewdata.m_isStaticView) {
    WorldViewWindow * worldviewWidget = dynamic_cast <WorldViewWindow *> ( this->parent());
        mainwindow = worldviewWidget->getMainWindow();
  }else{
    CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
    mainwindow = cameraViewWidget->getMainWindow();
  }

  if (mainwindow->getManipulator()){
    glLineWidth(1.0);
    mainwindow->getManipulator()->set_size(view->scale*mainwindow->getManipulator()->get_pivotSize());
    mainwindow->getManipulator()->draw();

    if (mainwindow->getManipulator()->get_movePivot()) {
      glLineStipple(1, 0);
      glDisable(GL_LINE_STIPPLE);
    }
  }
}

void GLView::enable_headlight()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);

    float position[4] = {0.0f,0.0f,0.0f,1.0f};
    glLightfv(GL_LIGHT0,GL_POSITION,position);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,1);

    float ambient[4] = {0.7f,0.7f,0.7f,1.0f};
    glMaterialfv(GL_FRONT,GL_AMBIENT,ambient);

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}



void GLView::draw_gradient(const float* top_color, const float* bot_color)
{
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glBegin(GL_QUADS);
    glColor3fv(bot_color);
    glVertex3i(-1,-1,-1);
    glVertex3i(1,-1,-1);
    glColor3fv(top_color);
    glVertex3i(1,1,-1);
    glVertex3i(-1,1,-1);
    glEnd();

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}

void GLView::draw_xz_grid(int width, int height, float scale)
{
    glPushAttrib(GL_LINE_BIT);

    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f,0.0f,0.0f);
    glVertex3f(-scale*width/2, 0.0f, 0.0f);
    glVertex3f(scale*width/2, 0.0f, 0.0f);
    glColor3f(0.0f,0.0f,1.0f);
    glVertex3f(0.0f, 0.0f, -scale*height/2);
    glVertex3f(0.0f, 0.0f, scale*height/2);
    glEnd();

    glColor3f(0.5f,0.5f,0.5f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int i = 0; i <= width; ++i) {
        glVertex3f(scale*(i-width/2), 0.0f, -scale*height/2);
        glVertex3f(scale*(i-width/2), 0.0f, scale*height/2);
    }
    for (int i = 0; i <= height; ++i) {
        glVertex3f(-scale*width/2, 0.0f, scale*(i-height/2));
        glVertex3f(scale*width/2, 0.0f, scale*(i-height/2));
    }
    glEnd();

    glPopAttrib();
}

void GLView::draw_cylinder(float radius, float height, int slices)
{
    for (int i = 0; i < slices; ++i) {
        float alpha = 2*M_PI*i/slices;
        float beta = 2*M_PI*(i+1)/slices;

        float cos_alpha = cos(alpha);
        float sin_alpha = sin(alpha);

        float cos_beta = cos(beta);
        float sin_beta = sin(beta);

        glBegin(GL_TRIANGLES);
        glNormal3f(0,-1,0);
        glVertex3f(radius*cos_alpha,-height/2,radius*sin_alpha);
        glVertex3f(radius*cos_beta,-height/2,radius*sin_beta);
        glVertex3f(0,-height/2,0);
        glEnd();

        glBegin(GL_QUADS);
        glNormal3f(cos_alpha,0,sin_alpha);
        glVertex3f(radius*cos_alpha,-height/2,radius*sin_alpha);
        glVertex3f(radius*cos_alpha,height/2,radius*sin_alpha);
        glVertex3f(radius*cos_beta,height/2,radius*sin_beta);
        glVertex3f(radius*cos_beta,-height/2,radius*sin_beta);
        glEnd();

        glBegin(GL_TRIANGLES);
        glNormal3f(0,1,0);
        glVertex3f(radius*cos_alpha,height/2,radius*sin_alpha);
        glVertex3f(0,height/2,0);
        glVertex3f(radius*cos_beta,height/2,radius*sin_beta);
        glEnd();
    }
}

void GLView::draw_camera()
{
    float length = 1.0f;
    float width = 0.3f;
    float height = 0.5f;

    glBegin(GL_QUADS);

    glNormal3f(0.0f,1.0f,0.0f);
    glVertex3f(-width,height,-length);
    glVertex3f(-width,height,length);
    glVertex3f(width,height,length);
    glVertex3f(width,height,-length);

    glNormal3f(1.0f,0.0f,0.0f);
    glVertex3f(width,-height,-length);
    glVertex3f(width,height,-length);
    glVertex3f(width,height,length);
    glVertex3f(width,-height,length);

    glNormal3f(0.0f,-1.0f,0.0f);
    glVertex3f(-width,-height,-length);
    glVertex3f(width,-height,-length);
    glVertex3f(width,-height,length);
    glVertex3f(-width,-height,length);

    glNormal3f(-1.0f,0.0f,0.0f);
    glVertex3f(-width,-height,-length);
    glVertex3f(-width,-height,length);
    glVertex3f(-width,height,length);
    glVertex3f(-width,height,-length);

    glNormal3f(0.0f,0.0f,1.0f);
    glVertex3f(-width,-height,length);
    glVertex3f(width,-height,length);
    glVertex3f(width,height,length);
    glVertex3f(-width,height,length);

    glNormal3f(0.0f,0.0f,-1.0f);
    glVertex3f(-width,-height,-length);
    glVertex3f(-width,height,-length);
    glVertex3f(width,height,-length);
    glVertex3f(width,-height,-length);

    glEnd();

    glBegin(GL_TRIANGLES);

    float mag = sqrt(height*height+9*length*length/25);

    glNormal3f(3*length/5/mag,0.0f,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(height,-height,-8*length/5);
    glVertex3f(height,height,-8*length/5);

    glNormal3f(-3*length/5/mag,0.0f,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(-height,height,-8*length/5);
    glVertex3f(-height,-height,-8*length/5);

    glNormal3f(0.0f,3*length/5/mag,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(height,height,-8*length/5);
    glVertex3f(-height,height,-8*length/5);

    glNormal3f(0.0f,-3*length/5/mag,height/mag);
    glVertex3f(0,0,-length);
    glVertex3f(-height,-height,-8*length/5);
    glVertex3f(height,-height,-8*length/5);

    glEnd();

    glPushMatrix();
    glTranslatef(0.0f,11*height/5,6*height/5);
    glRotatef(90.0f,0.0f,0.0f,1.0f);
    draw_cylinder(4*height/3,width,10);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0.0f,11*height/5,-6*height/5);
    glRotatef(90.0f,0.0f,0.0f,1.0f);
    draw_cylinder(4*height/3,width,10);
    glPopMatrix();
}

void GLView::draw_textured_quad(const double* pts, unsigned int texid)
{
    glPushAttrib(GL_ENABLE_BIT);

    //glColor3f(1.0f,1.0f,1.0f);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,texid);
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3d(pts[0], pts[1],  pts[2]);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3d(pts[3], pts[4],  pts[5]);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3d(pts[6], pts[7],  pts[8]);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3d(pts[9], pts[10], pts[11]);
    glEnd();

    glPopAttrib();
}
