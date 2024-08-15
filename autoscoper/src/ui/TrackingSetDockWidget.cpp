#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui_TrackingSetDockWidget.h"
#include "ui/TrackingSetDockWidget.h"
#include "ui/AutoscoperMainWindow.h"

#include "Tracker.hpp"
#include "Trial.hpp"

TrackingSetDockWidget::TrackingSetDockWidget(QWidget* parent)
  : QDockWidget(parent)
  , dock(new Ui::TrackingSetDockWidget)
{
  dock->setupUi(this);
  mainwindow = dynamic_cast<AutoscoperMainWindow*>(parent);
}

TrackingSetDockWidget::~TrackingSetDockWidget()
{
  delete dock;
}

void TrackingSetDockWidget::addTrackingSet(const int& num_volumes)
{
  this->tracking_sets.push_back("Tracking set " + std::to_string(dock->listWidget->count()));
  QListWidgetItem* item = new QListWidgetItem();
  int currentSet = this->mainwindow->getTracker()->trial()->numberOfCurveSets() - 1;
  item->setText(QString::fromStdString(this->tracking_sets[currentSet]));
  dock->listWidget->addItem(item);
  dock->listWidget->setCurrentItem(item);
}

void TrackingSetDockWidget::setCurrentSet(const int& idx)
{
  dock->listWidget->setCurrentRow(idx);
}

void TrackingSetDockWidget::on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* previous)
{
  if (current != NULL) {
    mainwindow->getTracker()->trial()->setCurrentCurveSet(dock->listWidget->row(current));
    mainwindow->redrawGL();
    mainwindow->frame_changed();
  }
}
