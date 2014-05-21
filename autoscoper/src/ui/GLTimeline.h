
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

public:
    GLTimeline(QWidget *parent);
	void setTrial(Trial* trial);
	void setGraphData(GraphData* position_graph);

private:
	Trial * m_trial;
	GraphData* m_position_graph;

	void draw_curve(const KeyCurve& curve);
};



#endif /* GLTIMELINE_H_ */
