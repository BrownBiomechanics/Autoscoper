#include "ui/TrackingOptionsDialog.h"
#include "ui_TrackingOptionsDialog.h"


TrackingOptionsDialog::TrackingOptionsDialog(QWidget *parent) :
												QDialog(parent),
												diag(new Ui::TrackingOptionsDialog){

	diag->setupUi(this);
}

TrackingOptionsDialog::~TrackingOptionsDialog(){
	delete diag;
}
