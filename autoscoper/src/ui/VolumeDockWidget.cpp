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

	// Store Model Name
	model_names_list.push_back(fi.completeBaseName().toStdString().c_str());

}

void VolumeDockWidget::on_treeWidget_currentItemChanged ( QTreeWidgetItem * current, QTreeWidgetItem * previous) {
	mainwindow->getTracker()->trial()->current_volume = dock->treeWidget->indexOfTopLevelItem(current);
	mainwindow->volume_changed();
}


QString VolumeDockWidget::getVolumeName(int volume_index) {
	std::string full_model_name = model_names_list[volume_index];
	size_t pos = full_model_name.find("_dcm_cropped");
	std::string model_name = full_model_name.substr(0, pos);
	QString selected_volume_name = QString::fromStdString(model_name);
	return selected_volume_name;
}