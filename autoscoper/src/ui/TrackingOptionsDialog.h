#ifndef TRACKINGOPTIONSDIALOG_H_
#define TRACKINGOPTIONSDIALOG_H_

#include <QDialog>


namespace Ui {
	class TrackingOptionsDialog;
}

class TrackingOptionsDialog : public QDialog{

	Q_OBJECT

	private:
		

	public:
		explicit TrackingOptionsDialog(QWidget *parent = 0);
		~TrackingOptionsDialog();

		Ui::TrackingOptionsDialog *diag;

	public slots:

};

#endif /* TRACKINGOPTIONSDIALOG_H_ */
