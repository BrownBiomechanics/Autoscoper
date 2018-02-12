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

/// \file ModelViewTreeWidgetItem.cpp
/// \author Benjamin Knorlein, Andy Loomis

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui/ModelViewTreeWidgetItem.h"
#include "ui/FilterTreeWidget.h"
#include "ui/FilterTreeWidgetItem.h"
#include "ui/CameraTreeWidgetItem.h"
#include "ui/FilterTreeWidgetParameter.h"

#include <QGridLayout>
#include <QToolButton>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QCheckBox>
#include <QSpacerItem>

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>       /* exp */

#include <View.hpp>
#ifdef WITH_CUDA
#include <gpu/cuda/RayCaster.hpp>
#else
#include <gpu/opencl/RayCaster.hpp>
#endif

ModelViewTreeWidgetItem::ModelViewTreeWidgetItem(int type, std::vector<Filter*>* filters):QTreeWidgetItem(MODEL_VIEW), QObject()
{
	m_filters = filters;
	m_type = type;

	switch(m_type){
		default:
		case 0:
			setName("Rad Renderer");
			break;

		case 1:
			setName("DRR Renderer");
			parameters.push_back(new FilterTreeWidgetParameter("Sample Distance",0.62,0.01,1.0,0.01));
			parameters.push_back(new FilterTreeWidgetParameter("XRay Intensity",0.49,0.0,1.0,0.01));
			parameters.push_back(new FilterTreeWidgetParameter("XRay Cutoff",0.00,0.0,1.0,0.01));
			break;
	}

	init();
}

void ModelViewTreeWidgetItem::save(std::ofstream & file){
	if(m_type == 0){
		file << "RadFilters_begin" << std::endl;
		for(int i = 0 ; i < childCount(); i ++){
			FilterTreeWidgetItem * filter = dynamic_cast<FilterTreeWidgetItem*> (child(i));
			if(filter){
				filter->save(file);
			}
		}
		file << "RadFilters_end" << std::endl;
	}else if (m_type == 1){
		file << "DrrRenderer_begin" << std::endl;
		file << "SampleDistance " << parameters[0]->value << std::endl;
		file << "RayIntensity " << parameters[1]->value << std::endl;
		file << "Cutoff " << parameters[2]->value << std::endl;
		file << "DrrRenderer_end" << std::endl;

		file << "DrrFilters_begin" << std::endl;
		for(int i = 0 ; i < childCount(); i ++){
			FilterTreeWidgetItem * filter = dynamic_cast<FilterTreeWidgetItem*> (child(i));
			if(filter){
				filter->save(file);
			}
		}
		file << "DrrFilters_end" << std::endl;
	}
}

void ModelViewTreeWidgetItem::loadSettings(std::ifstream & file){
	std::string line, key;
	if (m_type == 1){
		while (std::getline(file,line) && line.compare("DrrRenderer_end") != 0) {
            std::istringstream lineStream(line);
            lineStream >> key;
            if (key.compare("SampleDistance") == 0) {
                float value;
                lineStream >> value;
				parameters[0]->spinbox->setValue(value);
            }
            else if (key.compare("RayIntensity") == 0) {
                 float value;
                lineStream >> value;
                parameters[1]->spinbox->setValue(value);
            }
            else if (key.compare("Cutoff") == 0) {
                float value;
                lineStream >> value;
                parameters[2]->spinbox->setValue(value);
            }
        }
	}
}

void ModelViewTreeWidgetItem::loadFilters(std::ifstream & file){
	std::string line, key;

	//Delete all Filters
	for(int i = childCount() - 1; i >= 0 ; i --){
		FilterTreeWidgetItem * filterItem = dynamic_cast<FilterTreeWidgetItem*> (child(i)); 
		if(filterItem){
			removeFilter(filterItem);
			delete filterItem;	
		}
	}

	while (std::getline(file,line) && line.compare("DrrFilters_end") != 0  && line.compare("RadFilters_end") != 0 ) {
		std::istringstream lineStream(line);
		lineStream >> key;
		if (key.compare("SobelFilter_begin") == 0) {
			FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(0);
			filter->addToModelViewTreeWidgetItem(treeWidget(),this);
			filter->load(file);
		}
		else if (key.compare("ContrastFilter_begin") == 0) {
			FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(1);
			filter->addToModelViewTreeWidgetItem(treeWidget(),this);
			filter->load(file);
		}
		else if (key.compare("GaussianFilter_begin") == 0) {
			FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(2);
			filter->addToModelViewTreeWidgetItem(treeWidget(),this);
			filter->load(file);
		}
		else if (key.compare("SharpenFilter_begin") == 0) {
			FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(3);
			filter->addToModelViewTreeWidgetItem(treeWidget(),this);
			filter->load(file);
		}
	}
	
}


void ModelViewTreeWidgetItem::init(){
	settingsShown = false;

	pFrameSettings = NULL;
	settingsButton = NULL;
}

void ModelViewTreeWidgetItem::toggleVisible(){
	visibleCheckBox->toggle();
}

ModelViewTreeWidgetItem::~ModelViewTreeWidgetItem()
{
	if(settingsButton)delete settingsButton;
	if(pFrameSettings)delete pFrameSettings;
	delete visibleCheckBox;
	for(int i = 0; i < parameters.size(); i++){
		delete parameters[i];
	}
	parameters.clear();
} 

void ModelViewTreeWidgetItem::addToCameraTreeWidgetItem(QTreeWidget * treewidget, CameraTreeWidgetItem * cameraWidget){
	QFrame * pFrame = new QFrame(treewidget);
	pFrame->setMinimumHeight(32);
	QGridLayout* pLayout = new QGridLayout(pFrame);
	pLayout->addWidget(new QLabel(name), 0,1);
	visibleCheckBox = new QCheckBox();
	visibleCheckBox->setChecked(true);
	connect(visibleCheckBox,SIGNAL(stateChanged ( int )),this,SLOT(on_visibleCheckBox_stateChanged(int)));
	pLayout->addWidget(visibleCheckBox, 0,0);
	pLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum), 0,2);

	if(parameters.size() > 0){
		settingsButton = new QToolButton();
		QIcon icon;
		icon.addFile(QString::fromUtf8(":/images/resource-files/icons/settings.png"), QSize(), QIcon::Normal, QIcon::Off);
		settingsButton->setIcon(icon);
		settingsButton->setAutoRaise(true);
		settingsButton->connect(settingsButton, SIGNAL(clicked()),this,SLOT(settingsButtonClicked()));

		pLayout->addWidget(settingsButton, 0,3);

		pFrameSettings = new QFrame(pFrame);
		QGridLayout* pLayoutSettings = new QGridLayout(pFrameSettings);
		for(int i = 0; i < parameters.size(); i++){
			pLayoutSettings->addWidget(new QLabel(parameters[i]->name), i,0);
			QDoubleSpinBox * box = new QDoubleSpinBox();
			box->setMaximum(parameters[i]->maximumValue);
			box->setMinimum(parameters[i]->minimumValue);
			box->setSingleStep(parameters[i]->step);
			box->setValue(parameters[i]->value);
			parameters[i]->spinbox = box;
			connect(box,SIGNAL(valueChanged(double)),parameters[i],SLOT(valueChanged(double)));
			connect(parameters[i],SIGNAL(parameterChanged(void)),this,SLOT(updateModelview(void)));
			pLayoutSettings->addWidget(box, i,1);
		}
		pLayoutSettings->setMargin(0);
		pLayout->addWidget(pFrameSettings, 1,0,1,4);
		pFrameSettings->setHidden(!settingsShown);
		pLayout->setMargin(5);
	}
	else{
		pLayout->setMargin(5);
	}

	cameraWidget->addModelView(this);
    treewidget->setItemWidget(this, 0, pFrame);
	setExpanded(true);
	this->setBackgroundColor(0,QColor::fromRgb(240,240,240));
	this->setFlags(this->flags() & ~Qt::ItemIsDragEnabled);
}

void ModelViewTreeWidgetItem::settingsButtonClicked(){
	settingsShown = !settingsShown;
	pFrameSettings->setHidden(!settingsShown);

	if(settingsShown)
	{
		this->setBackgroundColor(0,QColor::fromRgb(225,225,225));
		QIcon icon;
		icon.addFile(QString::fromUtf8(":/images/resource-files/icons/settings_cancel.png"), QSize(), QIcon::Normal, QIcon::Off);
		settingsButton->setIcon(icon);
	}else{
		this->setBackgroundColor(0,QColor::fromRgb(240,240,240));
		QIcon icon;
		icon.addFile(QString::fromUtf8(":/images/resource-files/icons/settings.png"), QSize(), QIcon::Normal, QIcon::Off);
		settingsButton->setIcon(icon);
	}
	

	this->treeWidget()->doItemsLayout();
	this->treeWidget()->repaint();
}

void ModelViewTreeWidgetItem::addFilter(FilterTreeWidgetItem* filterItem, bool addToTree){
	filterTreeWidgets.push_back(filterItem);
	m_filters->push_back(filterItem->getFilter());
	if(addToTree)this->addChild(filterItem);
}

void ModelViewTreeWidgetItem::removeFilter(FilterTreeWidgetItem* filterItem, bool removeFromTree){
	m_filters->erase(std::remove(m_filters->begin(), m_filters->end(), filterItem->getFilter()), m_filters->end());
	filterTreeWidgets.erase(std::remove(filterTreeWidgets.begin(), filterTreeWidgets.end(), filterItem), filterTreeWidgets.end());
	if(removeFromTree)this->removeChild(filterItem);
}

void ModelViewTreeWidgetItem::printFilters(){
	for(int i = 0; i < filterTreeWidgets.size(); i++){
		fprintf(stderr,"          Filter%d %s\n",i,filterTreeWidgets[i]->getName().toStdString());
	}
}

void ModelViewTreeWidgetItem::resetVectors(){
	filterTreeWidgets.clear();
	m_filters->clear();
	for(int i = 0; i < childCount(); i++){
		FilterTreeWidgetItem * filter = dynamic_cast<FilterTreeWidgetItem*> (child(i));
		if(filter){
			filterTreeWidgets.push_back(filter);
			m_filters->push_back(filter->getFilter());
		}
	}
}

void ModelViewTreeWidgetItem::updateModelview(){
	if(m_type == 1){
		for (int idx = 0; idx < ((CameraTreeWidgetItem*)QTreeWidgetItem::parent())->getView()->nbDrrRenderer(); idx++){
			xromm::gpu::RayCaster* rayCaster = ((CameraTreeWidgetItem*)QTreeWidgetItem::parent())->getView()->drrRenderer(idx);  // not sure if this is correct. Dont we have to go through all drrRenderer

			double value = exp(7 * parameters[0]->value - 5);
			rayCaster->setSampleDistance(value);

			value = exp(15 * parameters[1]->value - 5);
			rayCaster->setRayIntensity(value);

			value = parameters[2]->value;
			value = value*(rayCaster->getMaxCutoff() - rayCaster->getMinCutoff()) + rayCaster->getMinCutoff();
			rayCaster->setCutoff(value);
		}
	}

	FilterTreeWidget * filterTreeWidget = dynamic_cast <FilterTreeWidget *>(treeWidget());
	filterTreeWidget->redrawGL();
}

void ModelViewTreeWidgetItem::on_visibleCheckBox_stateChanged ( int state ){
	if(state == 0){
		if(m_type == 0){
			((CameraTreeWidgetItem*) QTreeWidgetItem::parent())->getView()->rad_enabled = false;
		}else{
			((CameraTreeWidgetItem*) QTreeWidgetItem::parent())->getView()->drr_enabled = false;
		}
		
	}else{
		if(m_type == 0){
			((CameraTreeWidgetItem*) QTreeWidgetItem::parent())->getView()->rad_enabled = true;
		}else{
			((CameraTreeWidgetItem*) QTreeWidgetItem::parent())->getView()->drr_enabled = true;
		}
	}
	FilterTreeWidget * filterTreeWidget = dynamic_cast <FilterTreeWidget *>(treeWidget());
	filterTreeWidget->redrawGL();
}