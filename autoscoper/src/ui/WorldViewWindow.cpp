/*
 * ProgressDialog.cpp
 *
 *  Created on: Nov 19, 2013
 *      Author: ben
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui/WorldViewWindow.h"
#include "ui/AutoscoperMainWindow.h"

WorldViewWindow::WorldViewWindow(QWidget *parent):QDockWidget(parent)
{
    setWindowTitle(tr("World view"));
	
    openGL = new GLView(this);
	openGL->setStaticView(true);
	setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
	setAllowedAreas(Qt::AllDockWidgetAreas);
	setMinimumSize(0,0);
	setWindowFlags(windowFlags() & ~Qt::WindowStaysOnTopHint);
	this->resize(500,500);
    layout = new QGridLayout;
    layout->addWidget(openGL, 0, 0);
	setWidget(openGL);
	
	mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent);
}

void  WorldViewWindow::resizeEvent ( QResizeEvent * event )
{
	openGL->repaint();
}

void WorldViewWindow::setSharedGLContext(const QGLContext * sharedContext){
	QGLContext* context = new QGLContext(sharedContext->format(), openGL);
	context->create(sharedContext);
	openGL->setContext(context,sharedContext,true);
}

void WorldViewWindow::draw(){
	openGL->update();
}