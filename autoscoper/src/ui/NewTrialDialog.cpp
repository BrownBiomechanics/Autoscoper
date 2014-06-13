#include "ui/NewTrialDialog.h"
#include "ui_NewTrialDialog.h"
#include "ui/CameraBox.h"
#include "ui_CameraBox.h"

#include <QFileDialog>

#include <iostream>
//#include <sstream>
#include <stdexcept>


NewTrialDialog::NewTrialDialog(QWidget *parent) :
												QDialog(parent),
												diag(new Ui::NewTrialDialog){
	diag->setupUi(this);

	nbCams = 2;
	diag->label_CameraNb->setText(QString::number(nbCams));

	for (int i = 0; i < nbCams ; i ++){
		CameraBox* box = new CameraBox();
		box->widget->groupBox_Camera->setTitle("Camera "  + QString::number(i + 1));
		diag->gridLayout_6->addWidget(box, i, 0, 1, 1);
		cameras.push_back(box);
	}
}

NewTrialDialog::~NewTrialDialog(){
	delete diag;

	for (int i = 0; i < nbCams ; i ++){
		delete cameras[i];
	}
	cameras.clear();
}

void NewTrialDialog::on_toolButton_CameraMinus_clicked(){
	if(nbCams > 2){
		diag->gridLayout_6->removeWidget(cameras[nbCams-1]);
		delete cameras[nbCams-1];
		cameras.pop_back();
		
		nbCams -= 1;
		diag->label_CameraNb->setText(QString::number(nbCams));
	}
}


void NewTrialDialog::on_toolButton_CameraPlus_clicked(){
	nbCams += 1;
	diag->label_CameraNb->setText(QString::number(nbCams));

	CameraBox* box = new CameraBox();
	box->widget->groupBox_Camera->setTitle("Camera "  + QString::number(nbCams));
	diag->gridLayout_6->addWidget(box, nbCams - 1, 0, 1, 1);
	cameras.push_back(box);
}

void NewTrialDialog::on_toolButton_VolumeFile_clicked(){
	QString fileName = QFileDialog::getOpenFileName(this,
									tr("Open Volume File"), QDir::currentPath(),tr("Volume Object File (*.tif)"));
	if ( fileName.isNull() == false )
    {
		diag->lineEdit_VolumeFile->setText(fileName);
    }
}


void NewTrialDialog::on_pushButton_OK_clicked(){
	if(run()) this->accept();
}
void NewTrialDialog::on_pushButton_Cancel_clicked(){
	this->reject();
}

bool
NewTrialDialog::run()
{
	std::vector <QString> cameras_mayaCam;
	std::vector <QString> cameras_videoPath;

	for(int i = 0; i < nbCams; i++){
		if(!cameras[i]->widget->lineEdit_MayaCam->text().isEmpty()
			&& !cameras[i]->widget->lineEdit_VideoPath->text().isEmpty()){	
			cameras_mayaCam.push_back(cameras[i]->widget->lineEdit_MayaCam->text());
			cameras_videoPath.push_back(cameras[i]->widget->lineEdit_VideoPath->text());
		}else{
			return false;
		}
	}

	if(diag->lineEdit_VolumeFile->text().isEmpty()) return false;
	QString volume_filename = diag->lineEdit_VolumeFile->text();

	if(diag->lineEdit_ScaleX->text().isEmpty() || 
		diag->lineEdit_ScaleY->text().isEmpty() ||
			diag->lineEdit_ScaleZ->text().isEmpty() ) return false;

	double volume_scale_x = diag->lineEdit_ScaleX->text().toDouble();
    double volume_scale_y = diag->lineEdit_ScaleY->text().toDouble();
    double volume_scale_z = diag->lineEdit_ScaleZ->text().toDouble();

	int units = diag->comboBox_Units->currentIndex();

	switch (units) {
		case 0: { // micrometers->millimeters
			volume_scale_x /= 1000;
			volume_scale_y /= 1000;
			volume_scale_z /= 1000;
			break;
		}
		default:
		case 1: { // milimeters->millimeters
			break;
		}
		case 2: { // centemeters->millimeters
			volume_scale_x *= 10;
			volume_scale_y *= 10;
			volume_scale_z *= 10;
			break;
		}
	}

	bool volume_flip_x = diag->toolButton_FlipX->isChecked();
    bool volume_flip_y = diag->toolButton_FlipY->isChecked();
    bool volume_flip_z = diag->toolButton_FlipZ->isChecked();

    try {
        trial = xromm::Trial();
		int maxFrames = 0;
		for(int i = 0; i < nbCams; i++){
			trial.cameras.push_back(xromm::Camera(cameras_mayaCam[i].toStdString()));
			trial.videos.push_back(xromm::Video(cameras_videoPath[i].toStdString()));
        
			maxFrames = (maxFrames > trial.videos.at(i).num_frames()) ? maxFrames : trial.videos.at(i).num_frames() ;
		}


        trial.num_frames = maxFrames;

		xromm::Volume volume(volume_filename.toStdString());

        volume.scaleX(volume_scale_x);
        volume.scaleY(volume_scale_y);
        volume.scaleZ(volume_scale_z);

        volume.flipX(volume_flip_x);
        volume.flipY(volume_flip_y);
        volume.flipZ(volume_flip_z);

        trial.volumes.push_back(volume);

        trial.offsets[0] = 0.1;
        trial.offsets[1] = 0.1;
        trial.offsets[2] = 0.1;
        trial.offsets[3] = 0.1;
        trial.offsets[4] = 0.1;
        trial.offsets[5] = 0.1;

        trial.render_width = 512;
        trial.render_height = 512;

        return true;
    }
    catch ( std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
}