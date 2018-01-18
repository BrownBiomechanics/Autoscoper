#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui_VolumeDockWidget.h"
#include "ui/VolumeDockWidget.h"
#include "ui/AutoscoperMainWindow.h"

#include "Tracker.hpp"
#include "Trial.hpp"

#include <QFileInfo>
VolumeDockWidget::VolumeDockWidget(QWidget *parent) :
										QDockWidget(parent),
										dock(new Ui::VolumeDockWidget){
	dock->setupUi(this);

	mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent);
}

VolumeDockWidget::~VolumeDockWidget(){
	delete dock;
}

void VolumeDockWidget::clear(){
	dock->treeWidget->clear();
}

void VolumeDockWidget::addVolume(const std::string& filename){
	QTreeWidgetItem* volumeItem = new QTreeWidgetItem();
	QFileInfo fi (QString::fromStdString(filename));

	volumeItem->setText(0,fi.completeBaseName());
	dock->treeWidget->addTopLevelItem(volumeItem);
	dock->treeWidget->setCurrentItem(volumeItem);
}

void VolumeDockWidget::on_treeWidget_currentItemChanged ( QTreeWidgetItem * current, QTreeWidgetItem * previous) {
	mainwindow->getTracker()->trial()->current_volume = dock->treeWidget->indexOfTopLevelItem(current);
	mainwindow->volume_changed();
}