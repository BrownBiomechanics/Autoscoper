#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui/CameraTreeWidgetItem.h"
#include "ui/ModelViewTreeWidgetItem.h"
#include "ui/FilterTreeWidget.h"

#include <QGridLayout>
#include <QToolButton>
#include <QLabel>
#include <QFileInfo>

#include "View.hpp"
#include "Camera.hpp"

CameraTreeWidgetItem::CameraTreeWidgetItem(View * view):QTreeWidgetItem(CAMERA_VIEW), QObject()
{
	m_view = view;
	init();
}

CameraTreeWidgetItem::CameraTreeWidgetItem(View * view, QTreeWidget * parent):QTreeWidgetItem(parent, CAMERA_VIEW), QObject(parent)
{
	m_view = view;
	init();
}

void CameraTreeWidgetItem::init(){
	QFileInfo fi(m_view->camera()->mayacam().c_str());
	setName(fi.baseName());

	ModelViewTreeWidgetItem* rad = new ModelViewTreeWidgetItem(0, &m_view->radFilters());
	rad->addToCameraTreeWidgetItem(this->treeWidget(),this);
	
	ModelViewTreeWidgetItem* drr = new ModelViewTreeWidgetItem(1, &m_view->drrFilters());
	drr->addToCameraTreeWidgetItem(this->treeWidget(),this);
}

void CameraTreeWidgetItem::addToGrid(QTreeWidget * treewidget){
	QFrame* pFrame = new QFrame(treewidget);
	pFrame->setMinimumHeight(32);
    QGridLayout* pLayout = new QGridLayout(pFrame);
    pLayout->addWidget(new QLabel(name), 0,0);
	pLayout->setMargin(3);
	treewidget->addTopLevelItem(this);
    treewidget->setItemWidget(this, 0, pFrame);
	setExpanded(true);
	this->setBackgroundColor(0,QColor::fromRgb(230,230,230));
	this->setFlags(this->flags() & ~Qt::ItemIsDragEnabled & ~Qt::ItemIsDropEnabled);
}

CameraTreeWidgetItem::~CameraTreeWidgetItem()
{
} 

void CameraTreeWidgetItem::addModelView(ModelViewTreeWidgetItem* modelViewTtem){
	modelViewTreeWidgets.push_back(modelViewTtem);
	this->addChild(modelViewTtem);
}

void CameraTreeWidgetItem::removeModelView(ModelViewTreeWidgetItem* modelViewTtem){
	modelViewTreeWidgets.erase(std::remove(modelViewTreeWidgets.begin(), modelViewTreeWidgets.end(), modelViewTtem), modelViewTreeWidgets.end());
	this->removeChild(modelViewTtem);
}