
#ifndef GLTIMELINE_H_
#define GLTIMELINE_H_

#include <QGLWidget>
#include "ui/GLWidget.h"

namespace xromm {
	class Trial;
	
}
using xromm::Trial;
class KeyCurve;
struct GraphData;



class GLTimeline: public GLWidget
{
    Q_OBJECT

protected:
    void paintGL();
	void mousePressEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);

public:
    GLTimeline(QWidget *parent);
	void setTrial(Trial* trial);
	void setGraphData(GraphData* position_graph);

private:
	Trial * m_trial;
	GraphData* m_position_graph;

	bool draw_marquee;
	float marquee[4];
	bool modify_nodes;

	void mouse_to_graph(double mouse_x, double mouse_y, double& graph_x, double& graph_y);
	void draw_curve(const KeyCurve& curve);
};



#endif /* GLTIMELINE_H_ */
