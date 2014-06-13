
#ifndef GL_VIEW_H_
#define GL_VIEW_H_

#include <QGLWidget>
#include "ui/GLWidget.h"
#include "CoordFrame.hpp"

namespace xromm{
	namespace gpu{
		class View;
	}
}
using xromm::gpu::View;
using xromm::CoordFrame;

class QMouseEvent;
class QWheelEvent;

class GLView: public GLWidget
{
    Q_OBJECT

public:
    GLView(QWidget *parent);

	void setView(View * view);
	void setStaticView(bool staticView);

protected:
    void paintGL();
	void mousePressEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void wheelEvent(QWheelEvent *e);
private:
	View * m_view;
	CoordFrame * volume_matrix;
	
	// Default camera
	CoordFrame defaultViewMatrix;

	double press_x;
	double press_y;
	double prevx;
    double prevy;

	void select_manip_in_view(double x, double y, int button);
	void draw_manip_from_view(const ViewData* view);

	void move_manip_in_view(double x, double y, bool out_of_plane=false);
	void update_scale_in_view(ViewData* view);
	void enable_headlight();

	void draw_gradient(const float* top_color, const float* bot_color);
	void draw_xz_grid(int width, int height, float scale);
	void draw_cylinder(float radius, float height, int slices);
	void draw_camera();
	void draw_textured_quad(const double* pts, unsigned int texid);
};



#endif /* GL_VIEW_H_ */
