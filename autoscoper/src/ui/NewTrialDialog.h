#ifndef NEWTRIALDIALOG_H_
#define NEWTRIALDIALOG_H_

#include <QDialog>

namespace Ui {
	class NewTrialDialog;
}


#include "Trial.hpp"
using xromm::Trial;

class CameraBox;

class NewTrialDialog : public QDialog{

	Q_OBJECT

	private:
		std::vector <CameraBox *> cameras;	
		int nbCams;
		bool run();

	public:
		explicit NewTrialDialog(QWidget *parent = 0);
		~NewTrialDialog();

		Ui::NewTrialDialog *diag;

		Trial trial;
	
	public slots:

		void on_toolButton_CameraMinus_clicked();
		void on_toolButton_CameraPlus_clicked();

		void on_toolButton_VolumeFile_clicked();
		
		void on_pushButton_OK_clicked();
		void on_pushButton_Cancel_clicked();

};

#endif /* NEWTRIALDIALOG_H_ */
