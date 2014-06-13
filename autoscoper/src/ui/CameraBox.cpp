#include "ui/CameraBox.h"
#include "ui_CameraBox.h"

#include <QFileDialog>


CameraBox::CameraBox(QWidget *parent) :
												QWidget(parent),
												widget(new Ui::CameraBox){
	widget->setupUi(this);
}

CameraBox::~CameraBox(){
	delete widget;
}

		
void CameraBox::on_toolButton_MayaCam_clicked(){
	QString fileName = QFileDialog::getOpenFileName(this,
									tr("Open MayaCam File"), QDir::currentPath(),tr("MayaCam File (*.csv)"));
	if ( fileName.isNull() == false )
    {
		widget->lineEdit_MayaCam->setText(fileName);
    }
}
void CameraBox::on_toolButton_VideoPath_clicked(){
	QString inputPath = QFileDialog::getExistingDirectory (this,
									tr("Open VideoPath"), QDir::currentPath());
	if ( inputPath.isNull() == false )
    {
		widget->lineEdit_VideoPath->setText(inputPath);
    }
}
