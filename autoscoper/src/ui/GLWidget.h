#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>

// OpenGL error checking
#if defined GLDEBUG  
#define CALL_GL(exp) do{ \
    exp; \
    if (glGetError() != GL_NO_ERROR) \
        cerr << "Error in OpenGL call at " \
	         << __FILE__ << ':' << __LINE__ << endl; \
}while(0)
#else  
#define CALL_GL(exp) exp  
#endif  

struct ViewData
{
    int window_width;
    int window_height;

    float ratio;
    float fovy;
    float near_clip;
    float far_clip;

    float zoom;
    float zoom_x;
    float zoom_y;

    int viewport_x;
    int viewport_y;
    int viewport_width;
    int viewport_height;

    double scale;

    GLuint pbo;
};


class GLWidget: public QGLWidget
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent);

public slots:
    //void animate();
	

protected:
	void initializeGL();
	void resizeGL(int w, int h);

	//void mouseMoveEvent(QMouseEvent *e);

	//void wheelEvent(QWheelEvent *event);

	ViewData viewdata;
	
	void update_viewport(ViewData* view);

private:
	int w,h;
	
};



#endif /* GLWIDGET_H */
