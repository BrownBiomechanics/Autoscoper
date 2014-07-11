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

void ImportExportTrackingOptionsDialog::on_pushButton_OK_clicked(){
	this->accept();
}

void ImportExportTrackingOptionsDialog::on_pushButton_Cancel_clicked(){
	this->reject();
}