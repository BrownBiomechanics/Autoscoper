#ifndef VOLUMELISTWIDGETITEM_H
#define VOLUMELISTWIDGETITEM_H

#include <QListWidgetItem>
#include <QObject>

class AutoscoperMainWindow;
class QCheckBox;
namespace xromm {
  namespace gpu {
    class RayCaster;
  }
}

class VolumeListWidgetItem : public QObject, public QListWidgetItem {
  // List Object to display the volume name and a checkbox to toggle visibility
  Q_OBJECT
public:
  VolumeListWidgetItem(QListWidget* listWidget, const QString& name, AutoscoperMainWindow* main_window, std::vector< xromm::gpu::RayCaster*>* renderers);
private:
  std::vector< xromm::gpu::RayCaster*> renderers_;
  QCheckBox *visibilityCheckBox_;
  QString name_;
  AutoscoperMainWindow*  main_window_;

  void setup(QListWidget* listWidget);
public slots:
  void on_visiblilityCheckBox__toggled(bool checked);
};
#endif // !VOLUMELISTWIDGETITEM_H
