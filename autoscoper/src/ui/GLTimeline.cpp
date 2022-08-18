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

/// \file GLTimeline.cpp
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

#include "ui/GLTimeline.h"
#include "ui/TimelineDockWidget.h"
#include "ui/AutoscoperMainWindow.h"

#include "Trial.hpp"
#include "KeyCurve.hpp"
#include "Tracker.hpp"

#include <sstream>
#include <math.h>
#include <QMouseEvent>

GLTimeline::GLTimeline(QWidget *parent)
: GLWidget(parent)
{
	m_trial = NULL;
	m_position_graph = NULL;

	draw_marquee = false;
	modify_nodes = false;
}

void GLTimeline::setTrial(Trial* trial){
	m_trial = trial;
}

void GLTimeline::setGraphData(GraphData* position_graph){
	m_position_graph = position_graph;
}

// Renders a bitmap string at the specified position using glut.
void GLTimeline::render_bitmap_string(double x,
                                 double y,
                                 const char* string)
{
	setFont(QFont(this->font().family(), 10));
	QFontMetrics fm(this->font());
	renderText(x - fm.width(string) * 0.5, y, string);
}

void GLTimeline::renderText(double textPosX, double textPosY, QString text)
{
	QPainter painter(this);
	painter.setPen(Qt::yellow);
	painter.setFont(QFont("Helvetica", 10));
	painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
	painter.drawText(textPosX, textPosY, text); // z = pointT4.z + distOverOp / 4
	painter.end();
}

void GLTimeline::mouse_to_graph(double mouse_x,
               double mouse_y,
               double& graph_x,
               double& graph_y)
{
	TimelineDockWidget * timelineDockWidget = dynamic_cast <TimelineDockWidget *> ( this->parent()->parent());

	double frame_offset = 48.0*(timelineDockWidget->getPosition_graph()->max_frame-
                                timelineDockWidget->getPosition_graph()->min_frame)/
                                viewdata.viewport_width;
    double min_frame = timelineDockWidget->getPosition_graph()->min_frame-frame_offset;
    double max_frame = timelineDockWidget->getPosition_graph()->max_frame-1.0;
    double value_offset = 12.0*(timelineDockWidget->getPosition_graph()->max_value-
                               timelineDockWidget->getPosition_graph()->min_value)/
                                viewdata.viewport_height;
    double value_offset_top = 12.0*(timelineDockWidget->getPosition_graph()->max_value-
                                   timelineDockWidget->getPosition_graph()->min_value)/
                                   viewdata.viewport_height;
    double min_value = timelineDockWidget->getPosition_graph()->min_value-value_offset;
    double max_value = timelineDockWidget->getPosition_graph()->max_value+value_offset_top;

    double ndcx = mouse_x/viewdata.viewport_width;
    double ndcy = 1.0-mouse_y/viewdata.viewport_height;

    graph_x = min_frame+ndcx*(max_frame+1-min_frame);
    graph_y = min_value+ndcy*(max_value-min_value);
}

void GLTimeline::mousePressEvent(QMouseEvent *e){
	TimelineDockWidget * timelineDockWidget = dynamic_cast <TimelineDockWidget *> ( this->parent()->parent());
	AutoscoperMainWindow * mainwindow = timelineDockWidget->getMainWindow();

	// Only respond to left button click
    if (e->button() &  Qt::LeftButton)  {

        double x, y;
        mouse_to_graph(e->x(),e->y(),x,y);
        marquee[0] = x;
        marquee[1] = y;
        marquee[2] = x;
        marquee[3] = y;

        // If control is pressed then we are modifying nodes or tangents
        if ( Qt::ControlModifier & e->modifiers() ){
            modify_nodes = true;
            mainwindow->push_state();
        }

        // Otherwise we are creating a selection marquee
        else {
            draw_marquee = true;
        }
    }

    mainwindow->redrawGL();
}

void GLTimeline::mouseMoveEvent(QMouseEvent *e){
	TimelineDockWidget * timelineDockWidget = dynamic_cast <TimelineDockWidget *> ( this->parent()->parent());
	AutoscoperMainWindow * mainwindow = timelineDockWidget->getMainWindow();

	if (e->buttons() &  Qt::LeftButton)  {
        if (modify_nodes) {
            double x, y;
            mouse_to_graph(e->x(),e->y(),x,y);

            int dx = (int)(x-marquee[2]); // Clamp to integer values
            double dy = y-marquee[3];

            for (unsigned i = 0; i < timelineDockWidget->getSelectedNodes()->size(); i++) {
                KeyCurve& curve = *(*timelineDockWidget->getSelectedNodes())[i].first.first;
                KeyCurve::iterator it = (*timelineDockWidget->getSelectedNodes())[i].first.second;
                Selection_type type = (*timelineDockWidget->getSelectedNodes())[i].second;

				if (timelineDockWidget->getPosition_graph()->frame_locks.at((int)curve.time(it))) {
                    continue;
                }

                if (type == NODE) {
                    //node.set_x(node.get_x()+dx); // Prevent x from begin
                    //modified
                    curve.set_value(it,curve.value(it)+dy);
                }
                else if (type == IN_TANGENT) {
                    double in = curve.in_tangent(it)-dy;
                    curve.set_in_tangent(it,in);

                }
                else { // OUT_TANGENT
                    double out = curve.out_tangent(it)+dy;
                    curve.set_out_tangent(it,out);
                }
            }

            marquee[2] = abs(dx) > 0? x: marquee[2];
            marquee[3] = y;

            mainwindow->update_xyzypr_and_coord_frame();
        }
        else if (draw_marquee) {
            double x, y;
            mouse_to_graph(e->x(),e->y(),x,y);

            marquee[2] = x;
            marquee[3] = y;
        }
    }

    mainwindow->redrawGL();
}

void GLTimeline::mouseReleaseEvent(QMouseEvent *e){
	 // If there are selected nodes and
	TimelineDockWidget * timelineDockWidget = dynamic_cast <TimelineDockWidget *> ( this->parent()->parent());
	AutoscoperMainWindow * mainwindow = timelineDockWidget->getMainWindow();
    if (e->button() &  Qt::LeftButton)  {
        if (modify_nodes) {
            modify_nodes = false;
        }
        else if (draw_marquee) {

            float min_x = marquee[0] < marquee[2]? marquee[0]: marquee[2];
            float max_x = marquee[0] < marquee[2]? marquee[2]: marquee[0];
            float min_y = marquee[1] < marquee[3]? marquee[1]: marquee[3];
            float max_y = marquee[1] < marquee[3]? marquee[3]: marquee[1];

            double frame_offset = 48.0*(timelineDockWidget->getPosition_graph()->max_frame-
                                        timelineDockWidget->getPosition_graph()->min_frame)/
                                        viewdata.viewport_width;
            double min_frame = timelineDockWidget->getPosition_graph()->min_frame-frame_offset;
            double max_frame = timelineDockWidget->getPosition_graph()->max_frame-1.0;
            double value_offset = 12.0*(timelineDockWidget->getPosition_graph()->max_value-
                                        timelineDockWidget->getPosition_graph()->min_value)/
                                        viewdata.viewport_height;
            double value_offset_top = 12.0*(timelineDockWidget->getPosition_graph()->max_value-
                                           timelineDockWidget->getPosition_graph()->min_value)/
                                           viewdata.viewport_height;
            double min_value = timelineDockWidget->getPosition_graph()->min_value-value_offset;
            double max_value = timelineDockWidget->getPosition_graph()->max_value+value_offset_top;

            float a = (max_frame+1-min_frame)/(max_value-min_value)*
                       viewdata.viewport_height/viewdata.viewport_width;
            float tan_scale = 40.0f*(max_frame+1-min_frame)/viewdata.viewport_width;

            std::vector<std::pair<std::pair<KeyCurve*,KeyCurve::iterator>,Selection_type> > new_nodes;

			for (unsigned i = 0; i < timelineDockWidget->getSelectedNodes()->size(); i++) {
                KeyCurve& curve = *(*timelineDockWidget->getSelectedNodes())[i].first.first;
                KeyCurve::iterator it = (*timelineDockWidget->getSelectedNodes())[i].first.second;

                float s_in = tan_scale/sqrt(1.0f+a*a*curve.in_tangent(it)*curve.in_tangent(it));
                float s_out = tan_scale/sqrt(1.0f+a*a*curve.out_tangent(it)*curve.out_tangent(it));

                bool in_selected = curve.time(it)-s_in > min_x &&
                                   curve.time(it)-s_in < max_x &&
                                   curve.value(it)-s_in*curve.in_tangent(it) > min_y &&
                                   curve.value(it)-s_in*curve.in_tangent(it) < max_y;
                bool node_selected = curve.time(it) > min_x &&
                                     curve.time(it) < max_x &&
                                     curve.value(it) > min_y &&
                                     curve.value(it) < max_y;
                bool out_selected = curve.time(it)+s_out > min_x &&
                                    curve.time(it)+s_out < max_x &&
                                    curve.value(it)+s_out*curve.out_tangent(it) > min_y &&
                                    curve.value(it)+s_out*curve.out_tangent(it) < max_y;

                if (in_selected && !node_selected && !out_selected) {
                    new_nodes.push_back(make_pair(make_pair(&curve,it),IN_TANGENT));
                }
                else if (!in_selected && !node_selected && out_selected) {
                    new_nodes.push_back(make_pair(make_pair(&curve,it),OUT_TANGENT));
                }
            }

            //double v = 3.0;
            //double x_sense = (max_frame+1-min_frame)/viewdata.viewport_width;
            //double y_sense = (max_value-min_value)/viewdata.viewport_height;

            if (timelineDockWidget->getPosition_graph()->show_x) {
				KeyCurve::iterator it = mainwindow->getTracker()->trial()->getXCurve(-1)->begin();
				while (it != mainwindow->getTracker()->trial()->getXCurve(-1)->end()) {
					if (mainwindow->getTracker()->trial()->getXCurve(-1)->time(it) > min_x &&
						mainwindow->getTracker()->trial()->getXCurve(-1)->time(it) < max_x &&
						mainwindow->getTracker()->trial()->getXCurve(-1)->value(it) > min_y &&
						mainwindow->getTracker()->trial()->getXCurve(-1)->value(it) < max_y) {
						new_nodes.push_back(make_pair(make_pair(mainwindow->getTracker()->trial()->getXCurve(-1), it), NODE));
                    }
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_y) {
				KeyCurve::iterator it = mainwindow->getTracker()->trial()->getYCurve(-1)->begin();
				while (it != mainwindow->getTracker()->trial()->getYCurve(-1)->end()) {
					if (mainwindow->getTracker()->trial()->getYCurve(-1)->time(it) > min_x &&
						mainwindow->getTracker()->trial()->getYCurve(-1)->time(it) < max_x &&
						mainwindow->getTracker()->trial()->getYCurve(-1)->value(it) > min_y &&
						mainwindow->getTracker()->trial()->getYCurve(-1)->value(it) < max_y) {
						new_nodes.push_back(make_pair(make_pair(mainwindow->getTracker()->trial()->getYCurve(-1), it), NODE));
					}
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_z) {
				KeyCurve::iterator it = mainwindow->getTracker()->trial()->getZCurve(-1)->begin();
				while (it != mainwindow->getTracker()->trial()->getZCurve(-1)->end()) {
					if (mainwindow->getTracker()->trial()->getZCurve(-1)->time(it) > min_x &&
						mainwindow->getTracker()->trial()->getZCurve(-1)->time(it) < max_x &&
						mainwindow->getTracker()->trial()->getZCurve(-1)->value(it) > min_y &&
						mainwindow->getTracker()->trial()->getZCurve(-1)->value(it) < max_y) {
						new_nodes.push_back(make_pair(make_pair(mainwindow->getTracker()->trial()->getZCurve(-1), it), NODE));
					}
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_yaw) {
				KeyCurve::iterator it = mainwindow->getTracker()->trial()->getYawCurve(-1)->begin();
				while (it != mainwindow->getTracker()->trial()->getYawCurve(-1)->end()) {
					if (mainwindow->getTracker()->trial()->getYawCurve(-1)->time(it) > min_x &&
						mainwindow->getTracker()->trial()->getYawCurve(-1)->time(it) < max_x &&
						mainwindow->getTracker()->trial()->getYawCurve(-1)->value(it) > min_y &&
						mainwindow->getTracker()->trial()->getYawCurve(-1)->value(it) < max_y) {
						new_nodes.push_back(make_pair(make_pair(mainwindow->getTracker()->trial()->getYawCurve(-1), it), NODE));
					}
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_pitch) {
				KeyCurve::iterator it = mainwindow->getTracker()->trial()->getPitchCurve(-1)->begin();
				while (it != mainwindow->getTracker()->trial()->getPitchCurve(-1)->end()) {
					if (mainwindow->getTracker()->trial()->getPitchCurve(-1)->time(it) > min_x &&
						mainwindow->getTracker()->trial()->getPitchCurve(-1)->time(it) < max_x &&
						mainwindow->getTracker()->trial()->getPitchCurve(-1)->value(it) > min_y &&
						mainwindow->getTracker()->trial()->getPitchCurve(-1)->value(it) < max_y) {
						new_nodes.push_back(make_pair(make_pair(mainwindow->getTracker()->trial()->getPitchCurve(-1), it), NODE));
					}
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_roll) {
				KeyCurve::iterator it = mainwindow->getTracker()->trial()->getRollCurve(-1)->begin();
				while (it != mainwindow->getTracker()->trial()->getRollCurve(-1)->end()) {
					if (mainwindow->getTracker()->trial()->getRollCurve(-1)->time(it) > min_x &&
						mainwindow->getTracker()->trial()->getRollCurve(-1)->time(it) < max_x &&
						mainwindow->getTracker()->trial()->getRollCurve(-1)->value(it) > min_y &&
						mainwindow->getTracker()->trial()->getRollCurve(-1)->value(it) < max_y) {
						new_nodes.push_back(make_pair(make_pair(mainwindow->getTracker()->trial()->getRollCurve(-1), it), NODE));
					}
                    ++it;
                }
            }

			//fprintf(stderr,"Selected Nodes %zd %zd\n", new_nodes.size(), timelineDockWidget->getSelectedNodes()->size() );

			timelineDockWidget->setSelectedNodes(new_nodes);

            draw_marquee = false;
        }
    }

	mainwindow->redrawGL();

}

void GLTimeline::paintGL()
{
	if(m_position_graph){
		glPushAttrib(GL_ENABLE_BIT);
		glDisable(GL_DEPTH_TEST);

		glPushAttrib(GL_POINT_BIT);
		glPointSize(4.0);

		glPushAttrib(GL_LINE_BIT);
		glDisable(GL_LINE_SMOOTH);
		glLineWidth(1);

		// Calculate how much space needs to be left on the left of the
		// graph in order to accomodate the labels.
		double frame_offset = 48.0*(m_position_graph->max_frame-m_position_graph->min_frame)/
							  (double)viewdata.viewport_width;
		double min_frame = m_position_graph->min_frame-frame_offset;
		double max_frame = m_position_graph->max_frame-1.0;

        // Calculate how much space needs to be left on the bottom and top of the
        // graph in order to accomodate the labels.
		float value_offset = (float)12.0*(m_position_graph->max_value-m_position_graph->min_value)/
							  (float)viewdata.viewport_height;
		/*float value_offset_top = (float)12.0*(m_position_graph->max_value-m_position_graph->min_value)/
								  (float)viewdata.viewport_height;*/
		float min_value = (float)m_position_graph->min_value-value_offset;
		float max_value = (float)m_position_graph->max_value+value_offset;

        // Read the viewport
		glViewport(viewdata.viewport_x,
				   viewdata.viewport_y,
				   viewdata.viewport_width,
				   viewdata.viewport_height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(min_frame,max_frame+1,min_value,max_value);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		// Clear the buffers.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		double frame_dist = (int)ceil(frame_offset);
		//double value_dist = 3.0*value_offset;

		if (frame_dist < 1.0) {
			frame_dist = 1.0;
		}

		// Draw grid with grid lines separated by the above frame_dist and
		// value_dist distances. Those distances are calculated each time this
		// fucntion is called and are based on the size of the window.
		glColor3f(0.75f,0.75f,0.75f);

		// This section visualizes the x-axis grid lines (vertical grid lines)
		glBegin(GL_LINES);
		for (double x = m_position_graph->min_frame; x <= max_frame; x += frame_dist) {
			glVertex2d(x,min_value);
			glVertex2d(x,max_value);
		}
		glEnd();


        glColor3f(0.75f,0.75f,0.75f);
		// This section visualizes the y-axis grid lines (horizontal grid lines)
        //double grid_size = (max_value - min_value)/5;
        double mid_point = round_this((min_value + max_value + value_offset - value_offset)/2);
        std::vector<float> y_values;

        y_values.push_back(round_this(min_value+value_offset));
        y_values.push_back(round_this((mid_point+min_value+value_offset)/2));
        //y_values.push_back(round_this(mid_point));
        // In order to have equal spacing:
        y_values.push_back(round_this(y_values.at(1) + y_values.at(1) - y_values.at(0)));
        y_values.push_back(round_this(y_values.at(2) + y_values.at(2) - y_values.at(1)));
        y_values.push_back(round_this(y_values.at(3) + y_values.at(2) - y_values.at(1)));
        //y_values.push_back(round_this((mid_point+max_value-value_offset_top)/2));
        //y_values.push_back(round_this(max_value-value_offset_top));

		glBegin(GL_LINES);
		for (int counter = 0; counter < y_values.size(); counter++) {
			glVertex2d(min_frame,y_values.at(counter));
			glVertex2d(max_frame+1,y_values.at(counter));
		}
		/*for (double y = mid_point-grid_size; y > min_value; y -= grid_size) {
			glVertex2d(min_frame,y);
			glVertex2d(max_frame+1,y);
		}*/
		glEnd();

		// Draw the x and y reference coordinate system.
        glColor3f(0.0f,0.0f,0.0f);

		glBegin(GL_LINES);
		glVertex2d(min_frame,0.0);
		glVertex2d(max_frame+1,0.0);
		glVertex2d(0.0,min_value);
		glVertex2d(0.0,max_value);
		glEnd();


        // Draw grid labels.
        double char_width = (double)viewdata.viewport_width / (m_position_graph->max_frame - m_position_graph->min_frame + frame_offset);
        double char_height = (double)viewdata.viewport_height / (max_value - min_value);


        glLineWidth(1.5);
		glColor3f(0.0f,0.0f,0.0f);
		// This section visualizes the x-axis values
		for (double x = m_position_graph->min_frame; x <= max_frame; x += frame_dist) {
			std::stringstream ss; ss << (int)x;
			render_bitmap_string((x + frame_offset)* char_width, (double)viewdata.viewport_height - 2,
								 ss.str().c_str());
		}


        double diff = 0;
        for (int counter = 0; counter < y_values.size(); counter++) {
            std::stringstream ss; ss << y_values.at(counter);
            if (counter == 0)
            {
                render_bitmap_string(frame_offset * char_width * 0.5,
                (double)viewdata.viewport_height - 6,  ss.str().c_str());
            } else {
                diff = y_values.at(counter)-y_values.at(counter-1);
                render_bitmap_string(frame_offset * char_width * 0.5,
                (double)viewdata.viewport_height - diff * counter * char_height - 6,  ss.str().c_str());
            }
        }

		// This section visualizes the y-axis values
		/*for (double y = mid_point; y < max_value; y += value_offset) {
			std::stringstream ss; ss << (int)(y+0.5);
			render_bitmap_string(frame_offset* char_width * 0.5,
				y * char_height + (double)viewdata.viewport_height*0.5,
								 ss.str().c_str());
		}
		for (double y = mid_point; y > min_value-value_offset; y -= value_offset) {
			std::stringstream ss; ss << (int)(y+0.5);
			render_bitmap_string(frame_offset* char_width * 0.5,
				y * char_height + (double)viewdata.viewport_height*0.5,
								 ss.str().c_str());
		}*/

		// XXX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		if (draw_marquee) {

			glBegin(GL_LINES);
			glVertex2f(marquee[0],marquee[1]);
			glVertex2f(marquee[0],marquee[3]);

			glVertex2f(marquee[0],marquee[1]);
			glVertex2f(marquee[2],marquee[1]);

			glVertex2f(marquee[0],marquee[3]);
			glVertex2f(marquee[2],marquee[3]);

			glVertex2f(marquee[2],marquee[1]);
			glVertex2f(marquee[2],marquee[3]);
			glEnd();

			glEnable(GL_LINE_STIPPLE);
			glLineStipple(2,0x3333);

			glColor3f(1.0f,1.0f,1.0f);
			glBegin(GL_LINES);
			glVertex2f(marquee[0],marquee[1]);
			glVertex2f(marquee[0],marquee[3]);

			glVertex2f(marquee[0],marquee[1]);
			glVertex2f(marquee[2],marquee[1]);

			glVertex2f(marquee[0],marquee[3]);
			glVertex2f(marquee[2],marquee[3]);

			glVertex2f(marquee[2],marquee[1]);
			glVertex2f(marquee[2],marquee[3]);
			glEnd();

			glLineStipple(1,0);
			glDisable(GL_LINE_STIPPLE);
		}

		// Draw the key frame curves
		if(m_trial){
			// Draw current frame
			glColor3f(0.75f,0.75f,0.75f);
			glBegin(GL_LINES);
			glVertex2d((double)m_trial->frame,min_value);
			glVertex2d((double)m_trial->frame,max_value);
			glEnd();

			if (m_position_graph->show_x) {
				glColor3f(1.0f,0.0f,0.0f);
				draw_curve(*m_trial->getXCurve(-1));
			}

			if (m_position_graph->show_y) {
				glColor3f(0.0f,1.0f,0.0f);
				draw_curve(*m_trial->getYCurve(-1));
			}

			if (m_position_graph->show_z) {
				glColor3f(0.0f,0.0f,1.0f);
				draw_curve(*m_trial->getZCurve(-1));
			}

			if (m_position_graph->show_yaw) {
				glColor3f(1.0f,1.0f,0.0f);
				draw_curve(*m_trial->getYawCurve(-1));
			}

			if (m_position_graph->show_pitch) {
				glColor3f(1.0f,0.0f,1.0f);
				draw_curve(*m_trial->getPitchCurve(-1));
			}

			if (m_position_graph->show_roll) {
				glColor3f(0.0f,1.0f,1.0f);
				draw_curve(*m_trial->getRollCurve(-1));
			}
		}
		float a = (max_frame+1-min_frame)/(max_value-min_value)*
				  viewdata.viewport_height/viewdata.viewport_width;
		float tan_scale = 40.0f*(max_frame+1-min_frame)/viewdata.viewport_width;

		TimelineDockWidget * timelineDockWidget = dynamic_cast <TimelineDockWidget *> ( this->parent()->parent());

		for (unsigned i = 0; i < timelineDockWidget->getSelectedNodes()->size(); i++) {
			KeyCurve& curve = *(*timelineDockWidget->getSelectedNodes())[i].first.first;
			KeyCurve::iterator it = (*timelineDockWidget->getSelectedNodes())[i].first.second;
			Selection_type type = (*timelineDockWidget->getSelectedNodes())[i].second;

			float s_in = tan_scale/sqrt(1.0f+a*a*curve.in_tangent(it)*curve.in_tangent(it));
			float s_out = tan_scale/sqrt(1.0f+a*a*curve.out_tangent(it)*curve.out_tangent(it));

			glBegin(GL_LINES);

			if (type == NODE || type == IN_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
			else { glColor3f(0.0f,0.0f,0.0f); }

			glVertex2f(curve.time(it)-s_in,curve.value(it)-s_in*curve.in_tangent(it));
			glVertex2f(curve.time(it),curve.value(it));

			if (type == NODE || type == OUT_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
			else { glColor3f(0.0f,0.0f,0.0f); }

			glVertex2f(curve.time(it),curve.value(it));
			glVertex2f(curve.time(it)+s_out,curve.value(it)+s_out*curve.out_tangent(it));

			glEnd();

			glBegin(GL_POINTS);

			if (type == NODE || type == IN_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
			else { glColor3f(0.0f,0.0f,0.0f); }
			glVertex2f(curve.time(it)-s_in,curve.value(it)-s_in*curve.in_tangent(it));

			if (type == NODE) { glColor3f(1.0f,1.0f,0.0f); }
			else { glColor3f(0.0f,0.0f,0.0f); }
			glVertex2f(curve.time(it),curve.value(it));

			if (type == NODE || type == OUT_TANGENT) { glColor3f(1.0f,1.0f,0.0f); }
			else { glColor3f(0.0f,0.0f,0.0f); }
			glVertex2f(curve.time(it)+s_out,curve.value(it)+s_out*curve.out_tangent(it));

			glEnd();
		}

		glPopMatrix();
		glPopAttrib(); // GL_LINE_BIT
		glPopAttrib(); // GL_POINT_BIT
		glPopAttrib(); // GL_ENABLE_BIT
	}
}

void GLTimeline::draw_curve(const KeyCurve& curve)
{
    // Get the minimum and maximum x-values

    float min_x, max_x;
    KeyCurve::const_iterator it = curve.begin();
    if (it == curve.end()) {
        return;
    }

    min_x = curve.time(it);
    it = curve.end(); it--;
    max_x = curve.time(it);

    // Clamp the values to the extents of the graph

    if (min_x < m_position_graph->min_frame) {
        min_x = m_position_graph->min_frame;
    }

    if (max_x > m_position_graph->max_frame) {
        max_x = m_position_graph->max_frame;
    }

    // Calculate the number of curve segments to draw

    int num_segments = viewdata.window_width/8;
    float dx = (max_x-min_x)/num_segments;
    dx = 1.0f/(int)(1.0f+1.0f/dx);

    // Draw the curve

    glBegin(GL_LINE_STRIP);
    for (float x = min_x; x < max_x; x += dx) {
        glVertex2f(x,curve(x));
    }
    glVertex2f(max_x,curve(max_x));
    glEnd();

    // Draw the curve points

    glPushAttrib(GL_CURRENT_BIT);

    float current_color[4];
    glGetFloatv(GL_CURRENT_COLOR,current_color);

    glBegin(GL_POINTS);
    it = curve.begin();
    while (it != curve.end()) {
        if (curve.time(it) < min_x || curve.time(it) > max_x) {
            it++;
            continue;
        }

        if (m_position_graph->frame_locks.at((int)curve.time(it))) {
            glColor3fv(current_color);
        }
        else {
            glColor3f(0.0f,0.0f,0.0f);
        }

        glVertex2f(curve.time(it),curve.value(it));
        it++;
    }
    glEnd();

    glPopAttrib();
}

float GLTimeline::round_this(double my_val) {
    return floor(my_val*10+0.5f)/10;
}
