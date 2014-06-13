#ifndef OPENCLPLATFORMSELECTDIALOG_H_
#define OPENCLPLATFORMSELECTDIALOG_H_

#include <QDialog>

namespace Ui {
	class OpenCLPlatformSelectDialog;
}

class OpenCLPlatformSelectDialog : public QDialog{

	Q_OBJECT

	private:
		
	public:
		explicit OpenCLPlatformSelectDialog(QWidget *parent = 0);
		~OpenCLPlatformSelectDialog();

		Ui::OpenCLPlatformSelectDialog *diag;
		int getNumberPlatforms(){return platforms.size();}

	private:
		std::vector< std::vector<std::string> > platforms;

	public slots:
		void on_comboBox_currentIndexChanged ( int index );
		void on_pushButton_clicked();
};

#endif /* OPENCLPLATFORMSELECTDIALOG_H_ */
