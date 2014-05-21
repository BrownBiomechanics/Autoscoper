#ifndef IMPORTEXPORTTRACKINGOPTIONSDIALOG_H_
#define IMPORTEXPORTTRACKINGOPTIONSDIALOG_H_

#include <QDialog>

namespace Ui {
	class ImportExportTrackingOptionsDialog;
}

class ImportExportTrackingOptionsDialog : public QDialog{

	Q_OBJECT

	private:
		

	public:
		explicit ImportExportTrackingOptionsDialog(QWidget *parent = 0);
		~ImportExportTrackingOptionsDialog();

		Ui::ImportExportTrackingOptionsDialog *diag;

	public slots:
};

#endif /* IMPORTEXPORTTRACKINGOPTIONSDIALOG_H_ */
