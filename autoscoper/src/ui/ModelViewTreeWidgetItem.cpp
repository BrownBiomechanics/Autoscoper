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


//ModelViewTreeWidgetItem::ModelViewTreeWidgetItem(QString _name, std::vector<Filter*>* filters):QTreeWidgetItem(MODEL_VIEW), QObject()
//{
//	setName(_name);
//	m_filters = filters;
//
//	init();
//}
//
//ModelViewTreeWidgetItem::ModelViewTreeWidgetItem(QString _name, std::vector<Filter*>* filters, QTreeWidget * parent):QTreeWidgetItem(parent, MODEL_VIEW), QObject(parent)
//{
//	setName(_name);
//	m_filters = filters;
//
//	init();
//}

void ModelViewTreeWidgetItem::init(){
	settingsShown = false;

	pFrameSettings = NULL;
	settingsButton = NULL;
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
		fprintf(stderr,"          Filter%d %s\n",i,filterTreeWidgets[i]->getName().toStdString().c_str());
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
		xromm::gpu::RayCaster* rayCaster = ((CameraTreeWidgetItem*) QTreeWidgetItem::parent())->getView()->drrRenderer();

		double value = exp(7*parameters[0]->value-5);
		rayCaster->setSampleDistance(value);	

		value = exp(15*parameters[1]->value-5);
		rayCaster->setRayIntensity(value);	

		value = parameters[2]->value;
		value = value*(rayCaster->getMaxCutoff()-rayCaster->getMinCutoff()) +rayCaster->getMinCutoff();
		rayCaster->setCutoff(value);	
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