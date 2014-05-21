#ifndef GLTRACKER_H
#define GLTRACKER_H

#include <QGLWidget>

namespace xromm {
	class Trial;
	class Tracker;
}
using xromm::Tracker;

class GLTracker: public QGLWidget
{
    Q_OBJECT

public:
    GLTracker(Tracker * tracker , QWidget *parent = NULL);

public slots:
    //void animate();
	
protected:
    void paintGL();
	void initializeGL();

private:
	Tracker * m_tracker;
};



#endif /* GLTRACKER_H */
