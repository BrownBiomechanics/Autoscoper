#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui_VolumeDockWidget.h"
#include "ui/VolumeDockWidget.h"
#include "ui/AutoscoperMainWindow.h"
#include "ui/VolumeListWidgetItem.h"

#include "Tracker.hpp"
#include "Trial.hpp"
#include "View.hpp"

#include <QFileInfo>

#include <iostream>

VolumeDockWidget::VolumeDockWidget(QWidget* parent) :
  QDockWidget(parent),
  dock(new Ui::VolumeDockWidget)
{
  dock->setupUi(this);

  mainwindow  = dynamic_cast<AutoscoperMainWindow*> ( parent);
}

VolumeDockWidget::~VolumeDockWidget()
{
  delete dock;
}

void VolumeDockWidget::clearVol()
{
  model_names_list.clear();
  dock->listWidget->clear();

  /*int n = dock->listWidget->count();


     int counter = 0;
     while (n != 0)
     {
     n = dock->listWidget->count();
     delete(dock->listWidget->takeItem(0));
     counter++;
     }
     n = dock->listWidget->count();*/
}

void VolumeDockWidget::addVolume(const std::string& filename, int idx)
{
  QFileInfo fi (QString::fromStdString(filename));

  std::vector<xromm::gpu::RayCaster*> renderers = {};
  for (xromm::gpu::View* view : mainwindow->getTracker()->views()) {
    renderers.push_back(view->drrRenderer(idx));
  }

  QListWidgetItem* volumeItem = new VolumeListWidgetItem(dock->listWidget, fi.completeBaseName(), mainwindow, &renderers);
  dock->listWidget->addItem(volumeItem);
  dock->listWidget->setCurrentItem(volumeItem);

  // Store Model Name
  model_names_list.push_back(fi.completeBaseName().toStdString().c_str());

}

void VolumeDockWidget::setVolumeVisibility(unsigned int volume_index, bool visible)
{
  VolumeListWidgetItem* item = dynamic_cast<VolumeListWidgetItem*>(dock->listWidget->item(volume_index));
  item->setVisibility(visible);
}

void VolumeDockWidget::on_listWidget_currentItemChanged (QListWidgetItem* current, QListWidgetItem* previous)
{
  if (current != NULL) {
    mainwindow->getTracker()->trial()->current_volume = dock->listWidget->row(current);
    mainwindow->volume_changed();
  } else {
    mainwindow->getTracker()->trial()->current_volume = -1;
  }
}


QString VolumeDockWidget::getVolumeName(int volume_index)
{
  std::string full_model_name = model_names_list[volume_index];
  size_t pos = full_model_name.find("_dcm_cropped");
  std::string model_name = full_model_name.substr(0, pos);
  QString selected_volume_name = QString::fromStdString(model_name);
  return selected_volume_name;
}
