#ifndef CAMERABOX_H_
#define CAMERABOX_H_

#include <QWidget>

namespace Ui {
	class CameraBox;
}

class CameraBox : public QWidget{

	Q_OBJECT

	private:
		

	public:
		explicit CameraBox(QWidget *parent = 0);
		~CameraBox();

		Ui::CameraBox *widget;

	public slots:
		
		void on_toolButton_MayaCam_clicked();
		void on_toolButton_VideoPath_clicked();
};

#endif /* IMPORTEXPORTTRACKINGOPTIONSDIALOG_H_ */
