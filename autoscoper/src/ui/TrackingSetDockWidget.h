#ifndef TRACKINGSETDOCKWIDGET_H
#define TRACKINGSETDOCKWIDGET_H

#include <QDockWidget>

// forward declarations
namespace Ui {
class TrackingSetDockWidget;
}

class AutoscoperMainWindow;
class QListWidgetItem;

class TrackingSetDockWidget : public QDockWidget
{

  Q_OBJECT

public:
  explicit TrackingSetDockWidget(QWidget* parent = 0);
  ~TrackingSetDockWidget();

  AutoscoperMainWindow* getMainWindow() { return mainwindow; };

  void addTrackingSet(const int& num_volumes);
  void setCurrentSet(const int& idx);

private:
  Ui::TrackingSetDockWidget* dock;

  AutoscoperMainWindow* mainwindow;

  std::vector<std::string> tracking_sets;

protected:
public slots:
  void on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* previous);
};

#endif // TRACKINGSETDOCKWIDGET_H
