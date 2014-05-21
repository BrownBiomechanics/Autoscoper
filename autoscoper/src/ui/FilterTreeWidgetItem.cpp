#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui/FilterTreeWidgetItem.h"
#include "ui/ModelViewTreeWidgetItem.h"
#include "ui/FilterTreeWidget.h"
#include "ui/FilterTreeWidgetParameter.h"

#include <QGridLayout>
#include <QToolButton>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QCheckBox>

#ifdef WITH_CUDA
#include <gpu/cuda/SobelFilter.hpp>
#include <gpu/cuda/ContrastFilter.hpp>
#include <gpu/cuda/SharpenFilter.hpp>
#include <gpu/cuda/GaussianFilter.hpp>
#else
#include <gpu/opencl/SobelFilter.hpp>
#include <gpu/opencl/ContrastFilter.hpp>
#include <gpu/opencl/SharpenFilter.hpp>
#include <gpu/opencl/GaussianFilter.hpp>
#endif
#include <Filter.hpp>

FilterTreeWidgetItem::FilterTreeWidgetItem(int type):QTreeWidgetItem(FILTER), QObject()
{
	m_type = type;
	switch(type){
		default:
		case 0:
			setName("Sobel");
			parameters.push_back(new FilterTreeWidgetParameter("Blend",0.5,0.0,1.0,0.01));
			parameters.push_back(new FilterTreeWidgetParameter("Scale",1.0,0.0,10.0,0.1));
			m_filter = new xromm::gpu::SobelFilter();
			break;
		case 1:
			setName("Contrast");
			parameters.push_back(new FilterTreeWidgetParameter("Alpha",1.0,0.0,10.0,0.1));
			parameters.push_back(new FilterTreeWidgetParameter("Beta",1.0,0.0,10.0,0.1));
			m_filter = new xromm::gpu::ContrastFilter();
			break;
		case 2:
			setName("Gaussian");
			parameters.push_back(new FilterTreeWidgetParameter("Radius",1.0,0.0,10.0,0.1));
			m_filter = new xromm::gpu::GaussianFilter();
			break;
		case 3:
			setName("Sharpen");
			parameters.push_back(new FilterTreeWidgetParameter("Radius",1.0,0.0,10.0,0.1));
			parameters.push_back(new FilterTreeWidgetParameter("Contrast",1.0,0.0,10.0,0.1));
			m_filter = new xromm::gpu::SharpenFilter();
			break;
	}
	settingsShown = false;
	init();


}

//FilterTreeWidgetItem::FilterTreeWidgetItem(QString _name):QTreeWidgetItem(FILTER), QObject()
//{
//	setName(_name);
//	settingsShown = false;
//}
//
//FilterTreeWidgetItem::FilterTreeWidgetItem(QString _name, QTreeWidget * parent):QTreeWidgetItem(parent, FILTER), QObject(parent)
//{
//	setName(_name);
//	settingsShown = false;
//}

void FilterTreeWidgetItem::init(){
	settingsShown = false;

	pFrameSettings = NULL;
	settingsButton = NULL;
}

FilterTreeWidgetItem::~FilterTreeWidgetItem()
{
	delete m_filter;
	delete settingsButton;
	delete pFrameSettings;
	delete visibleCheckBox;
	for(int i = 0; i < parameters.size(); i++){
		delete parameters[i];
	}
	parameters.clear();
} 

void FilterTreeWidgetItem::addToModelViewTreeWidgetItem(QTreeWidget * treewidget, ModelViewTreeWidgetItem * modelViewWidget, bool addToTree){
	
	QFrame *pFrame = new QFrame(treewidget);
	pFrame->setMinimumHeight(32);
	QGridLayout* pLayout = new QGridLayout(pFrame);
	pLayout->addWidget(new QLabel(name), 0,1);
	visibleCheckBox = new QCheckBox();
	visibleCheckBox->setChecked(m_filter->enabled());
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
			connect(parameters[i],SIGNAL(parameterChanged(void)),this,SLOT(updateFilter(void)));
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

	modelViewWidget->addFilter(this,addToTree);
    treewidget->setItemWidget(this, 0, pFrame);
	setExpanded(true);
	this->setFlags(this->flags() & ~Qt::ItemIsDropEnabled);
}

void FilterTreeWidgetItem::settingsButtonClicked(){
	settingsShown = !settingsShown;
	pFrameSettings->setHidden(!settingsShown);

	if(settingsShown)
	{
		this->setBackgroundColor(0,QColor::fromRgb(235,235,235));
		QIcon icon;
		icon.addFile(QString::fromUtf8(":/images/resource-files/icons/settings_cancel.png"), QSize(), QIcon::Normal, QIcon::Off);
		settingsButton->setIcon(icon);
	}else{
		this->setBackgroundColor(0,QColor::fromRgb(255,255,255));
		QIcon icon;
		icon.addFile(QString::fromUtf8(":/images/resource-files/icons/settings.png"), QSize(), QIcon::Normal, QIcon::Off);
		settingsButton->setIcon(icon);
	}
	
	this->treeWidget()->doItemsLayout();
	this->treeWidget()->repaint();
}

void FilterTreeWidgetItem::updateFilter(){
	double value;
	switch(m_type){
		default:
		case 0:
			((xromm::gpu::SobelFilter*) (m_filter))->setBlend(parameters[0]->value);
			((xromm::gpu::SobelFilter*) (m_filter))->setScale(parameters[1]->value);
			break;
		case 1:
			((xromm::gpu::ContrastFilter*) (m_filter))->set_alpha(parameters[0]->value);
			((xromm::gpu::ContrastFilter*) (m_filter))->set_beta(parameters[1]->value);
			break;
		case 2:
			((xromm::gpu::GaussianFilter*) (m_filter))->set_radius(parameters[0]->value);
			break;
		case 3:
			((xromm::gpu::SharpenFilter*) (m_filter))->set_radius(parameters[0]->value);
			((xromm::gpu::SharpenFilter*) (m_filter))->set_contrast(parameters[1]->value);
			break;
	}
	FilterTreeWidget * filterTreeWidget = dynamic_cast <FilterTreeWidget *>(treeWidget());
	filterTreeWidget->redrawGL();
}

void FilterTreeWidgetItem::on_visibleCheckBox_stateChanged ( int state ){
	if(state == 0){
		m_filter->set_enabled(false);
	}else{
		m_filter->set_enabled(true);
	}
	FilterTreeWidget * filterTreeWidget = dynamic_cast <FilterTreeWidget *>(treeWidget());
	filterTreeWidget->redrawGL();
}
