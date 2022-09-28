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

/// \file GLTracker.cpp
/// \author Benjamin Knorlein, Andy Loomis

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>
#include <iostream>

#include "ui/GLTracker.h"
#include "Tracker.hpp"
//#include <QOpenGLContext>
//#include <QOpenGLFunctions>

#ifdef WITH_CUDA
#else
#include <gpu/opencl/OpenCL.hpp>
#endif

GLTracker::GLTracker(Tracker * tracker , QWidget *parent)
  : QOpenGLWidget(parent)
{
  //m_tracker = tracker;
    setAutoFillBackground(false);
  //makeCurrent();
  show();
  initializeGL();
  //shared_context = new QOpenGLContext(this);
  //context()->setShareContext(shared_context);
  //shared_context->create();
  hide();
}

GLTracker::~GLTracker()
{
  // delete shared_context;
}

void GLTracker::initializeGL(){
  glewInit();

  std::cout << "Graphics Card Vendor: "<< glGetString(GL_VENDOR) << std::endl;
  std::cout << glGetString(GL_RENDERER) << std::endl;
  std::cout << glGetString(GL_VERSION)  << std::endl;

  //m_tracker->init();

  std::cerr << "Initializing OpenGL..." << std::endl;


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

//void GLTracker::paintGL()
//{

// }
