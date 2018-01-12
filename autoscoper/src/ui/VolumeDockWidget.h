#ifndef VOLUMEDOCKWIDGET_H
#define VOLUMEDOCKWIDGET_H

#include <QDockWidget>

//forward declarations
namespace Ui {
	class VolumeDockWidget;
}

class AutoscoperMainWindow;
class QTreeWidgetItem;

class VolumeDockWidget : public QDockWidget{

	Q_OBJECT

	public:
		explicit VolumeDockWidget(QWidget *parent = 0);
		~VolumeDockWidget();

		AutoscoperMainWindow * getMainWindow(){return mainwindow;};
		
		void clear();
		void addVolume(const std::string& filename);

	private:
		Ui::VolumeDockWidget *dock;

		AutoscoperMainWindow * mainwindow;

	protected:

	public slots:
		void on_treeWidget_currentItemChanged ( QTreeWidgetItem * current, QTreeWidgetItem * previous);
};

#endif  // UAUTOSCOPERMAINWINDOW_H
