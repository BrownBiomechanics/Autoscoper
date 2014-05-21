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

#include "Trial.hpp"
#include "KeyCurve.hpp"

#include <sstream>

GLTimeline::GLTimeline(QWidget *parent)
    : GLWidget(parent)
{
	m_trial = NULL;
	m_position_graph = NULL;
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

		/*if (draw_marquee) {

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
		}*/

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

		/*for (unsigned i = 0; i < selected_nodes.size(); i++) {
			KeyCurve& curve = *selected_nodes[i].first.first;
			KeyCurve::iterator it = selected_nodes[i].first.second;
			Selection_type type = selected_nodes[i].second;

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
		}*/

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