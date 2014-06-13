
//#include "ui/AutoscoperMainWindow.h"
#include "ui/OpenCLPlatformSelectDialog.h"
#include "ui_OpenCLPlatformSelectDialog.h"

#ifndef WITH_CUDA
#include <gpu/opencl/OpenCL.hpp>
#endif

#include <QString>

OpenCLPlatformSelectDialog::OpenCLPlatformSelectDialog(QWidget *parent) :
												QDialog(parent),
												diag(new Ui::OpenCLPlatformSelectDialog){

	diag->setupUi(this);

#ifndef WITH_CUDA
	platforms = xromm::gpu::get_platforms();
	for (int i = 0 ; i < platforms.size();i++){
		diag->comboBox->addItem(QString::fromStdString(platforms[i][0]));
	}
#endif
}

OpenCLPlatformSelectDialog::~OpenCLPlatformSelectDialog(){
	delete diag;
}

void OpenCLPlatformSelectDialog::on_comboBox_currentIndexChanged ( int index ){
	
	QString text = "";
	for (int i = 0 ; i < platforms[index].size();i++){
		text = text + QString::fromStdString(platforms[index][i]) + '\n';
	}

	diag->textedit->setText(text);
}

void OpenCLPlatformSelectDialog::on_pushButton_clicked(){
#ifndef WITH_CUDA
	xromm::gpu::setUsedPlatform(diag->comboBox->currentIndex());
#endif
	this->close();
}