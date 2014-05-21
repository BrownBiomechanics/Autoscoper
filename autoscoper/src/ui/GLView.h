
#ifndef GL_VIEW_H_
#define GL_VIEW_H_

#include <QGLWidget>
#include "ui/GLWidget.h"

namespace xromm{
	namespace gpu{
		class View;
	}
	class CoordFrame;
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

protected:
    void paintGL();
	void mousePressEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void wheelEvent(QWheelEvent *e);
private:
	View * m_view;
	CoordFrame * volume_matrix;

	double press_x;
	double press_y;
	double prevx;
    double prevy;

	void select_manip_in_view(double x, double y, int button);
	void draw_manip_from_view(const ViewData* view);

	void move_manip_in_view(double x, double y, bool out_of_plane=false);
	void update_scale_in_view(ViewData* view);
	void enable_headlight();
};



#endif /* GL_VIEW_H_ */
