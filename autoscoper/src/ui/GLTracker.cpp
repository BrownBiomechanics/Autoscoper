#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>

#include "ui/GLTracker.h"
#include "Tracker.hpp"
#ifdef WITH_CUDA
#else
#include <gpu/opencl/OpenCL.hpp>
#endif

GLTracker::GLTracker(Tracker * tracker , QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	m_tracker = tracker;
    setAutoFillBackground(false);
	makeCurrent();
	initializeGL();
}


void GLTracker::initializeGL(){
	std::cout << "Graphics Card Vendor"<< glGetString(GL_VENDOR)   << std::endl;
	std::cout << glGetString(GL_RENDERER) << std::endl;
	std::cout << glGetString(GL_VERSION)  << std::endl;

	m_tracker->init();

	std::cerr << "Initializing OpenGL..." << std::endl;

	glewInit();

    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.5,0.5,0.5,1.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

#ifdef WITH_CUDA
#else
	std::cerr << "Initializing OpenCL-OpenGL interoperability..." << std::endl;
	xromm::gpu::opencl_global_gl_context();
#endif
}

void GLTracker::paintGL()
{

}
