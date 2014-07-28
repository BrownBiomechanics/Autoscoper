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
void render_bitmap_string(double x,
                                 double y,
                                 void* font,
                                 const char* string)
{
    glRasterPos2d(x,y);
    for (const char* c = string; *c != '\0'; ++c) {
        glutBitmapCharacter(font, *c);
    }
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
    double value_offset = 24.0*(timelineDockWidget->getPosition_graph()->max_value-
                               timelineDockWidget->getPosition_graph()->min_value)/
                                viewdata.viewport_height;
    double value_offset_top = 8.0*(timelineDockWidget->getPosition_graph()->max_value-
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
            double value_offset = 24.0*(timelineDockWidget->getPosition_graph()->max_value-
                                        timelineDockWidget->getPosition_graph()->min_value)/
                                        viewdata.viewport_height;
            double value_offset_top = 8.0*(timelineDockWidget->getPosition_graph()->max_value-
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
				KeyCurve::iterator it = mainwindow->getTracker()->trial()->x_curve.begin();
                while (it != mainwindow->getTracker()->trial()->x_curve.end()) {
                    if (mainwindow->getTracker()->trial()->x_curve.time(it) > min_x &&
                        mainwindow->getTracker()->trial()->x_curve.time(it) < max_x &&
                        mainwindow->getTracker()->trial()->x_curve.value(it) > min_y &&
                        mainwindow->getTracker()->trial()->x_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&mainwindow->getTracker()->trial()->x_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_y) {
                KeyCurve::iterator it = mainwindow->getTracker()->trial()->y_curve.begin();
                while (it != mainwindow->getTracker()->trial()->y_curve.end()) {
                    if (mainwindow->getTracker()->trial()->y_curve.time(it) > min_x &&
                        mainwindow->getTracker()->trial()->y_curve.time(it) < max_x &&
                        mainwindow->getTracker()->trial()->y_curve.value(it) > min_y &&
                        mainwindow->getTracker()->trial()->y_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&mainwindow->getTracker()->trial()->y_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_z) {
                KeyCurve::iterator it = mainwindow->getTracker()->trial()->z_curve.begin();
                while (it != mainwindow->getTracker()->trial()->z_curve.end()) {
                    if (mainwindow->getTracker()->trial()->z_curve.time(it) > min_x &&
                        mainwindow->getTracker()->trial()->z_curve.time(it) < max_x &&
                        mainwindow->getTracker()->trial()->z_curve.value(it) > min_y &&
                        mainwindow->getTracker()->trial()->z_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&mainwindow->getTracker()->trial()->z_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_yaw) {
                KeyCurve::iterator it = mainwindow->getTracker()->trial()->yaw_curve.begin();
                while (it != mainwindow->getTracker()->trial()->yaw_curve.end()) {
                    if (mainwindow->getTracker()->trial()->yaw_curve.time(it) > min_x &&
                        mainwindow->getTracker()->trial()->yaw_curve.time(it) < max_x &&
                        mainwindow->getTracker()->trial()->yaw_curve.value(it) > min_y &&
                        mainwindow->getTracker()->trial()->yaw_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&mainwindow->getTracker()->trial()->yaw_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_pitch) {
                KeyCurve::iterator it = mainwindow->getTracker()->trial()->pitch_curve.begin();
                while (it != mainwindow->getTracker()->trial()->pitch_curve.end()) {
                    if (mainwindow->getTracker()->trial()->pitch_curve.time(it) > min_x &&
                        mainwindow->getTracker()->trial()->pitch_curve.time(it) < max_x &&
                        mainwindow->getTracker()->trial()->pitch_curve.value(it) > min_y &&
                        mainwindow->getTracker()->trial()->pitch_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&mainwindow->getTracker()->trial()->pitch_curve,it),NODE));
                    }
                    ++it;
                }
            }
            if (timelineDockWidget->getPosition_graph()->show_roll) {
                KeyCurve::iterator it = mainwindow->getTracker()->trial()->roll_curve.begin();
                while (it != mainwindow->getTracker()->trial()->roll_curve.end()) {
                    if (mainwindow->getTracker()->trial()->roll_curve.time(it) > min_x &&
                        mainwindow->getTracker()->trial()->roll_curve.time(it) < max_x &&
                        mainwindow->getTracker()->trial()->roll_curve.value(it) > min_y &&
                        mainwindow->getTracker()->trial()->roll_curve.value(it) < max_y) {
                        new_nodes.push_back(make_pair(make_pair(&mainwindow->getTracker()->trial()->roll_curve,it),NODE));
                    }
                    ++it;
                }
            }

			fprintf(stderr,"Selected Nodes %d %d\n", new_nodes.size(), timelineDockWidget->getSelectedNodes()->size() );

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
		glPointSize(3.0);

		glPushAttrib(GL_LINE_BIT);
		glDisable(GL_LINE_SMOOTH);
		glLineWidth(1.0);

		// Calculate how much space needs to be left on the left and bottom of the
		// graph in order to accomodate the labels.
		double frame_offset = 48.0*(m_position_graph->max_frame-m_position_graph->min_frame)/
							  viewdata.viewport_width;
		double min_frame = m_position_graph->min_frame-frame_offset;
		double max_frame = m_position_graph->max_frame-1.0;
		double value_offset = 24.0*(m_position_graph->max_value-m_position_graph->min_value)/
							  viewdata.viewport_height;
		double value_offset_top = 8.0*(m_position_graph->max_value-m_position_graph->min_value)/
								  viewdata.viewport_height;
		double min_value = m_position_graph->min_value-value_offset;
		double max_value = m_position_graph->max_value+value_offset_top;

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
		double value_dist = 3.0*value_offset;

		if (frame_dist < 1.0) {
			frame_dist = 1.0;
		}

		// Draw grid with grid lines separated by the above frame_dist and
		// value_dist distances. Those distances are calculated each time this
		// fucntion is called and are based on the size of the window.
		glColor3f(0.25f,0.25f,0.25f);
		glBegin(GL_LINES);
		for (double x = m_position_graph->min_frame; x <= max_frame; x += frame_dist) {
			glVertex2d(x,min_value);
			glVertex2d(x,max_value);
		}
		glEnd();
		glBegin(GL_LINES);
		for (double y = 0; y < max_value; y += value_dist) {
			glVertex2d(min_frame,y);
			glVertex2d(max_frame+1,y);
		}
		for (double y = 0; y > min_value; y -= value_dist) {
			glVertex2d(min_frame,y);
			glVertex2d(max_frame+1,y);
		}
		glEnd();

		// Draw the x and y axes.
		glColor3f(0.75f,0.75f,0.75f);
		glBegin(GL_LINES);
		glVertex2d(min_frame,0.0);
		glVertex2d(max_frame+1,0.0);
		glVertex2d(0.0,min_value);
		glVertex2d(0.0,max_value);
		glEnd();

		// Draw grid labels.
		double char_width = 8.0*(m_position_graph->max_frame-m_position_graph->min_frame-frame_offset)/
							viewdata.viewport_width;
		double char_height = 13.0*(m_position_graph->max_value-m_position_graph->min_value-value_offset)/
							 viewdata.viewport_height;

		glColor3f(0.0f,0.0f,0.0f);
		for (double x = m_position_graph->min_frame; x <= max_frame; x += frame_dist) {
			std::stringstream ss; ss << (int)x;
			render_bitmap_string(x-char_width*ss.str().length()/2.0,
								 min_value+char_height/2.0,
								 GLUT_BITMAP_8_BY_13,
								 ss.str().c_str());
		}
		for (double y = 0; y < max_value; y += value_dist) {
			std::stringstream ss; ss << (int)(y+0.5);
			render_bitmap_string(min_frame+char_width/2.0,
								 y-char_height/2.0,
								 GLUT_BITMAP_8_BY_13,
								 ss.str().c_str());
		}
		for (double y = 0; y > min_value-value_offset; y -= value_dist) {
			std::stringstream ss; ss << (int)(y+0.5);
			render_bitmap_string(min_frame+char_width/2.0,
								 y-char_height/2.0,
								 GLUT_BITMAP_8_BY_13,
								 ss.str().c_str());
		}

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
				draw_curve(m_trial->x_curve);
			}

			if (m_position_graph->show_y) {
				glColor3f(0.0f,1.0f,0.0f);
				draw_curve(m_trial->y_curve);
			}

			if (m_position_graph->show_z) {
				glColor3f(0.0f,0.0f,1.0f);
				draw_curve(m_trial->z_curve);
			}

			if (m_position_graph->show_yaw) {
				glColor3f(1.0f,1.0f,0.0f);
				draw_curve(m_trial->yaw_curve);
			}

			if (m_position_graph->show_pitch) {
				glColor3f(1.0f,0.0f,1.0f);
				draw_curve(m_trial->pitch_curve);
			}

			if (m_position_graph->show_roll) {
				glColor3f(0.0f,1.0f,1.0f);
				draw_curve(m_trial->roll_curve);
			}
		}
		float a = (max_frame+1-min_frame)/(max_value-min_value)*
				  viewdata.viewport_height/viewdata.viewport_width;
		float tan_scale = 40.0f*(max_frame+1-min_frame)/viewdata.viewport_width;

		TimelineDockWidget * timelineDockWidget = dynamic_cast <TimelineDockWidget *> ( this->parent()->parent());
		//AutoscoperMainWindow * mainwindow = timelineDockWidget->getMainWindow();
		
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