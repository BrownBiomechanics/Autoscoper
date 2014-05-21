/*
 * Image_subwindow.cpp
 *
 *  Created on: Nov 19, 2013
 *      Author: ben
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui/CameraViewWidget.h"
#include "ui_CameraViewWidget.h"
#include "ui/AutoscoperMainWindow.h"

#include <QGLContext>

CameraViewWidget::CameraViewWidget(int id, View * view, QString name,QWidget *parent) :
											QWidget(parent),
												widget(new Ui::CameraViewWidget){
	widget->setupUi(this);
	m_name = name;
	m_id = id;
	widget->cameraTitleLabel->setText(m_name);
	widget->glView->setView(view);
	mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent);
}

CameraViewWidget::~CameraViewWidget(){

}

void CameraViewWidget::setSharedGLContext(const QGLContext * sharedContext){
	QGLContext* context = new QGLContext(sharedContext->format(), widget->glView);
	context->create(sharedContext);
	widget->glView->setContext(context,sharedContext,true);
}

void CameraViewWidget::draw(){
	widget->glView->update();
}