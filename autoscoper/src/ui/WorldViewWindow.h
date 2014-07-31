/*
 * ProgressDialog.h
 *
 *  Created on: Nov 19, 2013
 *      Author: ben
 */

#ifndef WORLDVIEWWINDOW_H_
#define WORLDVIEWWINDOW_H_

#include "ui/GLView.h"
#include <QGridLayout>
#include <QDockWidget>

class AutoscoperMainWindow;

class WorldViewWindow : public QDockWidget
{
    Q_OBJECT

public:
    WorldViewWindow(QWidget *parent);
	GLView *openGL;
	void setSharedGLContext(const QGLContext * sharedContext);

	AutoscoperMainWindow * getMainWindow(){return mainwindow;};
	void draw();

private:
	QGridLayout *layout;

	AutoscoperMainWindow * mainwindow;
protected:
	void  resizeEvent ( QResizeEvent * event );
};



#endif /* WORLDVIEWWINDOW_H_ */
