#include "ui/VolumeListWidgetItem.h"
#include "ui/AutoscoperMainWindow.h"

#include <QCheckBox>
#include <QDockWidget>
#include <QLabel>
#include <QGridLayout>
#include <QSpacerItem>

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
#include <gpu/cuda/RayCaster.hpp>
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
#include <gpu/opencl/RayCaster.hpp>
#endif

VolumeListWidgetItem::VolumeListWidgetItem(QListWidget* listWidget,const QString& name, AutoscoperMainWindow* main_window, std::vector< xromm::gpu::RayCaster*>* renderers) : QListWidgetItem(listWidget) {
  this->name_ = name;
  this->main_window_ = main_window;
  for (xromm::gpu::RayCaster* renderer : *renderers) {
    renderers_.push_back(renderer);
  }
  setup(listWidget);
}

void VolumeListWidgetItem::setup(QListWidget* listWidget) {
  // add a layout
  QFrame* pFrame = new QFrame(listWidget);
  pFrame->setMinimumHeight(32);
  QGridLayout* pLayout = new QGridLayout(pFrame);
  pLayout->addWidget(new QLabel(name_), 0, 1);
  visibilityCheckBox_ = new QCheckBox();
  visibilityCheckBox_->setChecked(true);
  pLayout->addWidget(visibilityCheckBox_, 0, 0);
  pLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum), 0, 2);
  pLayout->setMargin(5);
  pFrame->setLayout(pLayout);
  listWidget->setItemWidget(this, pFrame);
  QObject::connect(visibilityCheckBox_, SIGNAL(toggled(bool)), this, SLOT(on_visiblilityCheckBox__toggled(bool)));
}

void VolumeListWidgetItem::setVisibility(bool visible) {
  visibilityCheckBox_->setChecked(visible);
}

void VolumeListWidgetItem::on_visiblilityCheckBox__toggled(bool checked) {
  if (renderers_.size() > 0) {
    for (xromm::gpu::RayCaster* renderer : renderers_) {
      renderer->setVisible(checked);
    }
  }
  main_window_->volume_changed(); // update the volume
}