#ifndef VOLUMEDOCKWIDGET_H
#define VOLUMEDOCKWIDGET_H

#include <QDockWidget>

//forward declarations
namespace Ui {
  class VolumeDockWidget;
}

class AutoscoperMainWindow;
class QListWidgetItem;

class VolumeDockWidget : public QDockWidget{

  Q_OBJECT

  public:
    explicit VolumeDockWidget(QWidget *parent = 0);
    ~VolumeDockWidget();

    AutoscoperMainWindow * getMainWindow(){return mainwindow;};

    void clearVol();
    void addVolume(const std::string& filename);


    QString getVolumeName(int volume_index);
    QString getFullVolumeName(int volume_index);

  private:
    Ui::VolumeDockWidget *dock;

    AutoscoperMainWindow * mainwindow;

    std::vector<std::string> model_names_list;

  protected:

  public slots:
    void on_listWidget_currentItemChanged( QListWidgetItem * current, QListWidgetItem* previous);

};

#endif  // UAUTOSCOPERMAINWINDOW_H
