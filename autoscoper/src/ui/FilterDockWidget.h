#ifndef FILTERDOCKWIDGET_H
#define FILTERDOCKWIDGET_H

#include <QDockWidget>

//forward declarations
namespace Ui {
	class FilterDockWidget;
}
namespace xromm{
	namespace gpu{
		class View;
	}
}
using xromm::gpu::View;
class AutoscoperMainWindow;

class FilterDockWidget : public QDockWidget{

	Q_OBJECT

	public:
		explicit FilterDockWidget(QWidget *parent = 0);
		~FilterDockWidget();

		void clearTree();
		void addCamera(View * view);
		void toggle_drrs();
		void saveAllSettings(QString directory);
		void loadAllSettings(QString directory);

		AutoscoperMainWindow * getMainWindow(){return mainwindow;};

	private:
		Ui::FilterDockWidget *dock;

		AutoscoperMainWindow * mainwindow;

	protected:

	public slots:
};

#endif  // UAUTOSCOPERMAINWINDOW_H
