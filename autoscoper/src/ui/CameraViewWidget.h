/*
 * OptionsVisualizationDialog.h
 *
 *  Created on: Nov 19, 2013
 *      Author: ben
 */

#ifndef CAMERAVIEWWIDGET_H
#define CAMERAVIEWWIDGET_H

#include <QWidget>

class QGLContext;

//forward declarations
namespace Ui {
	class CameraViewWidget;
}
class Camera;

namespace xromm{
	namespace gpu{
		class View;
	}
}
using xromm::gpu::View;

class AutoscoperMainWindow;

class CameraViewWidget : public QWidget{

	Q_OBJECT

	public:
		explicit CameraViewWidget(int id, View * view, QString _name, QWidget *parent = 0);
		~CameraViewWidget();

		Ui::CameraViewWidget *widget;

		void setSharedGLContext(const QGLContext * sharedContext);

		int getID(){return m_id;};

		AutoscoperMainWindow * getMainWindow(){return mainwindow;};

		void draw();

	protected:

	public slots:

	private:
		QString m_name;
		int m_id;

		AutoscoperMainWindow * mainwindow;
};

#endif  // CAMERAVIEWWIDGET_H
