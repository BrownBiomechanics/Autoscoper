#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#ifdef _WIN32
  #include <windows.h>
#endif
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "ui/GLView.h"
#include "ui/AutoscoperMainWindow.h"
#include "ui/CameraViewWidget.h"
#include "ui/TimelineDockWidget.h"

#include "Tracker.hpp"
#include "View.hpp"
#include "CoordFrame.hpp"
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
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());

    // Setup the view from this perspective so that we can simply call set_view
    // on the manipulator
    glViewport(viewdata.viewport_x,
               viewdata.viewport_y,
               viewdata.viewport_width,
               viewdata.viewport_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(viewdata.fovy,viewdata.ratio,viewdata.near_clip,viewdata.far_clip);

	CoordFrame viewMatrix = cameraViewWidget->getMainWindow()->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame();

    double m[16];
    viewMatrix.inverse().to_matrix(m);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(m);

    cameraViewWidget->getMainWindow()->getManipulator()->set_view();
    cameraViewWidget->getMainWindow()->getManipulator()->set_size(viewdata.scale*cameraViewWidget->getMainWindow()->getManipulator()->get_pivotSize());
    cameraViewWidget->getMainWindow()->getManipulator()->on_mouse_press(x,viewdata.window_height-y);
}

 void GLView::wheelEvent(QWheelEvent *e)
 {
	 CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());

	 if ( Qt::ControlModifier & e->modifiers() ) {
		 if (e->delta() > 0) {
            viewdata.zoom *= 1.1f;
        }
        else if (e->delta() < 0) {
            viewdata.zoom /= 1.1f;
        }

        update_viewport(&viewdata);
        cameraViewWidget->getMainWindow()->redrawGL();
    }
 }

void GLView::mousePressEvent(QMouseEvent *e){
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());

	press_x = e->x();
    press_y = e->y();
	prevx = e->x();
	prevy = e->y();

    select_manip_in_view(e->x(),e->y(),e->button());

	cameraViewWidget->getMainWindow()->redrawGL();
}

// Moves the manipulator and volume based on the view, the selected axis, and
// the direction of the motion.
void GLView::move_manip_in_view(double x, double y, bool out_of_plane)
{
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
	AutoscoperMainWindow * mainwindow = cameraViewWidget->getMainWindow();

	if (mainwindow->getPosition_graph()->frame_locks.at(mainwindow->getTracker()->trial()->frame)) {
        return;
    }

    CoordFrame frame;
	if (mainwindow->getManipulator()->get_movePivot()) {
        frame = (CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()))* *mainwindow->getVolume_matrix());
    }

    if (!out_of_plane) {
		mainwindow->getManipulator()->set_size(viewdata.scale*mainwindow->getManipulator()->get_pivotSize());
        mainwindow->getManipulator()->on_mouse_move(x,viewdata.window_height-y);
    }
    else if (mainwindow->getManipulator()->selection() == Manip3D::VIEW_PLANE) {
		CoordFrame mmat = CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()));
		CoordFrame viewMatrix = mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame();

        double zdir[3] = { mmat.translation()[0]-viewMatrix.translation()[0],
                           mmat.translation()[1]-viewMatrix.translation()[1],
                           mmat.translation()[2]-viewMatrix.translation()[2]};
        double mag = sqrt(zdir[0]*zdir[0]+zdir[1]*zdir[1]+zdir[2]*zdir[2]);
        zdir[0] /= mag;
        zdir[1] /= mag;
        zdir[2] /= mag;

        double ztrans[3] = { (x-y)/2.0*zdir[0],(x-y)/2.0*zdir[1],(x-y)/2.0*zdir[2] };

        mmat.translate(ztrans);

		double m[16];
		mmat.to_matrix_row_order(m);
		mainwindow->getManipulator()->set_transform(Mat4d(m));

        mainwindow->getManipulator()->set_selection(Manip3D::VIEW_PLANE);
    }
	
	if (mainwindow->getManipulator()->get_movePivot()) {
        CoordFrame new_manip_matrix = CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()));
		mainwindow->setVolume_matrix(new_manip_matrix.inverse()*frame);
    }
}

void GLView::mouseMoveEvent(QMouseEvent *e){
	double dx = e->x() - prevx;
    double dy = e->y() - prevy;

	double x = e->x();
	double y = e->y();
	if ( Qt::ControlModifier & e->modifiers() ) {
		if (e->buttons() &  Qt::LeftButton) {
            viewdata.zoom_x -= dx/200/viewdata.zoom;
            viewdata.zoom_y += dy/200/viewdata.zoom;

            update_viewport(&viewdata);
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
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
	AutoscoperMainWindow * mainwindow = cameraViewWidget->getMainWindow();

    mainwindow->update_xyzypr();

    prevx = x;
    prevy = y;
}

void GLView::mouseReleaseEvent(QMouseEvent *e){
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
   
	cameraViewWidget->getMainWindow()->getManipulator()->on_mouse_release(e->x(),e->y());

    cameraViewWidget->getMainWindow()->update_graph_min_max(cameraViewWidget->getMainWindow()->getTracker()->trial()->frame);

	cameraViewWidget->getMainWindow()->redrawGL();
}

void GLView::paintGL()
{
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
	AutoscoperMainWindow * mainwindow = cameraViewWidget->getMainWindow();

	if(mainwindow && cameraViewWidget){
		update_scale_in_view(&viewdata);
		update_viewport(&viewdata);
		
		glViewport(viewdata.viewport_x,
				   viewdata.viewport_y,
				   viewdata.viewport_width,
				   viewdata.viewport_height);

		double m[16];
		CoordFrame modelview  = mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().inverse()
								* CoordFrame::from_matrix(trans(mainwindow->getManipulator()->transform()))
							    * (*mainwindow->getVolume_matrix());

		double imv[16];
		modelview.inverse().to_matrix_row_order(imv);
		m_view->drrRenderer()->setInvModelView(imv);

		float temp = 2.0f*sqrt(5.0)*sin(M_PI*viewdata.fovy/360.0);
		float width = temp/viewdata.zoom, height = temp/viewdata.zoom;
		float x = viewdata.zoom_x-width/2.0f, y = viewdata.zoom_y-height/2.0f;

		m_view->drrRenderer()->setViewport(
			viewdata.ratio*x, y, viewdata.ratio*width, height);
		m_view->radRenderer()->set_viewport(
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
		CALL_GL(glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, view->pbo));
		CALL_GL(glDrawPixels(view->window_width,
						view->window_height,
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

void GLView::update_scale_in_view(ViewData* view)
{
    // Determine the distance from the center of the pivot point to the
    // center of the view.
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
	AutoscoperMainWindow * mainwindow = cameraViewWidget->getMainWindow();
	
	CoordFrame mat = CoordFrame::from_matrix(trans(cameraViewWidget->getMainWindow()->getManipulator()->transform()));
    
	double dist_vec[3];
	dist_vec[0] = mat.translation()[0]-
        mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().translation()[0];
    dist_vec[1] = mat.translation()[1]-
        mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().translation()[1];
    dist_vec[2] = mat.translation()[2]-
        mainwindow->getTracker()->trial()->cameras.at(cameraViewWidget->getID()).coord_frame().translation()[2];
    
    double dist = sqrt(dist_vec[0]*dist_vec[0]+
                       dist_vec[1]*dist_vec[1]+
                       dist_vec[2]*dist_vec[2]);

    // Adjust the size of the pivot based on the distance.
    view->scale = 2.0*dist*tan(view->fovy*M_PI/360.0)*view->near_clip/view->zoom;
}

void GLView::draw_manip_from_view(const ViewData* view)
{
	CameraViewWidget * cameraViewWidget = dynamic_cast <CameraViewWidget *> ( this->parent());
   
	if ( cameraViewWidget->getMainWindow()->getManipulator()->get_movePivot()) {
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2,0x3333);
    }
	
	glLineWidth(1.0);
	cameraViewWidget->getMainWindow()->getManipulator()->set_size(view->scale*cameraViewWidget->getMainWindow()->getManipulator()->get_pivotSize());
    cameraViewWidget->getMainWindow()->getManipulator()->draw();

   if ( cameraViewWidget->getMainWindow()->getManipulator()->get_movePivot()) {
        glLineStipple(1,0);
        glDisable(GL_LINE_STIPPLE);
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


