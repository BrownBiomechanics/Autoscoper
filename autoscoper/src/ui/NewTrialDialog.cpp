// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

/// \file NewTrialDialog.cpp
/// \author Benjamin Knorlein, Andy Loomis

#include "ui/NewTrialDialog.h"
#include "ui_NewTrialDialog.h"
#include "ui/CameraBox.h"
#include "ui_CameraBox.h"
#include "ui/VolumeBox.h"
#include "ui_VolumeBox.h"

#include <QFileDialog>

#include <iostream>
//#include <sstream>
#include <stdexcept>


NewTrialDialog::NewTrialDialog(QWidget *parent) :
												QDialog(parent),
												diag(new Ui::NewTrialDialog){
	diag->setupUi(this);

	nbCams = 1;
	diag->label_CameraNb->setText(QString::number(nbCams));

	for (int i = 0; i < nbCams; i++){
		CameraBox* box = new CameraBox();
		box->widget->groupBox_Camera->setTitle("Camera " + QString::number(i + 1));
		diag->gridLayout_6->addWidget(box, i, 0, 1, 1);
		cameras.push_back(box);
	}

	nbVolumes = 1;
	diag->label_VolumeNb->setText(QString::number(nbVolumes));

	for (int i = 0; i < nbVolumes; i++){
		VolumeBox* box = new VolumeBox();
		box->widget->groupBox_Volume->setTitle("Volume " + QString::number(i + 1));
		diag->gridLayout_8->addWidget(box, i, 0, 1, 1);
		volumes.push_back(box);
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
	if(nbCams > 1){
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

void NewTrialDialog::on_toolButton_VolumeMinus_clicked()
{
	if (nbCams >= 1 && nbVolumes!=0){
        
		diag->gridLayout_8->removeWidget(volumes[nbVolumes - 1]);
		delete volumes[nbVolumes - 1];
		volumes.pop_back();

        nbVolumes -= 1;
        
		diag->label_VolumeNb->setText(QString::number(nbVolumes));
	}
}

void NewTrialDialog::on_toolButton_VolumePlus_clicked()
{
	nbVolumes += 1;
	diag->label_VolumeNb->setText(QString::number(nbVolumes));

	VolumeBox* box = new VolumeBox();
	box->widget->groupBox_Volume->setTitle("Volume " + QString::number(nbVolumes));
	diag->gridLayout_8->addWidget(box, nbVolumes - 1, 0, 1, 1);
	volumes.push_back(box);
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

    try {
        trial = xromm::Trial();
		int maxFrames = 0;
		for(int i = 0; i < nbCams; i++){
			trial.cameras.push_back(xromm::Camera(cameras_mayaCam[i].toStdString().c_str()));
			trial.videos.push_back(xromm::Video(cameras_videoPath[i].toStdString().c_str()));
        
			maxFrames = (maxFrames > trial.videos.at(i).num_frames()) ? maxFrames : trial.videos.at(i).num_frames() ;
		}

        trial.num_frames = maxFrames;


		for (int i = 0; i < nbVolumes; i++){
			if (volumes[i]->widget->lineEdit_VolumeFile->text().isEmpty())
				continue;

			QString volume_filename = volumes[i]->widget->lineEdit_VolumeFile->text();

			if (volumes[i]->widget->lineEdit_ScaleX->text().isEmpty() ||
				volumes[i]->widget->lineEdit_ScaleY->text().isEmpty() ||
				volumes[i]->widget->lineEdit_ScaleZ->text().isEmpty()) 
					continue;

			double volume_scale_x = volumes[i]->widget->lineEdit_ScaleX->text().toDouble();
			double volume_scale_y = volumes[i]->widget->lineEdit_ScaleY->text().toDouble();
			double volume_scale_z = volumes[i]->widget->lineEdit_ScaleZ->text().toDouble();

			int units = volumes[i]->widget->comboBox_Units->currentIndex();

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

			bool volume_flip_x = volumes[i]->widget->toolButton_FlipX->isChecked();
			bool volume_flip_y = volumes[i]->widget->toolButton_FlipY->isChecked();
			bool volume_flip_z = volumes[i]->widget->toolButton_FlipZ->isChecked();

			xromm::Volume volume(volume_filename.toStdString().c_str());

			volume.scaleX(volume_scale_x);
			volume.scaleY(volume_scale_y);
			volume.scaleZ(volume_scale_z);

			volume.flipX(volume_flip_x);
			volume.flipY(volume_flip_y);
			volume.flipZ(volume_flip_z);

			trial.volumes.push_back(volume);
			trial.volumestransform.push_back(xromm::VolumeTransform());
			trial.num_volumes += 1;
		}

        trial.offsets[0] = 0.1;
        trial.offsets[1] = 0.1;
        trial.offsets[2] = 0.1;
        trial.offsets[3] = 0.1;
        trial.offsets[4] = 0.1;
        trial.offsets[5] = 0.1;

        trial.render_width = 880;
        trial.render_height = 880;

        return true;
    }
    catch ( std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
}