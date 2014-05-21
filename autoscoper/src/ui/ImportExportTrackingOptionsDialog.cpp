#include "ui/ImportExportTrackingOptionsDialog.h"
#include "ui_ImportExportTrackingOptionsDialog.h"

ImportExportTrackingOptionsDialog::ImportExportTrackingOptionsDialog(QWidget *parent) :
												QDialog(parent),
												diag(new Ui::ImportExportTrackingOptionsDialog){
	diag->setupUi(this);
}

ImportExportTrackingOptionsDialog::~ImportExportTrackingOptionsDialog(){
	delete diag;
}
