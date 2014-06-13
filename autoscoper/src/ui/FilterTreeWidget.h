#ifndef FILTERTREEWIDGET_H
#define FILTERTREEWIDGET_H

#include <QTreeWidget>

static const int CAMERA_VIEW = QTreeWidgetItem::UserType;
static const int MODEL_VIEW = QTreeWidgetItem::UserType + 1;
static const int FILTER = QTreeWidgetItem::UserType + 2;

namespace xromm{
	namespace gpu{
		class View;
	}
}
using xromm::gpu::View;

class FilterTreeWidget : public QTreeWidget{

	Q_OBJECT

	public:
		explicit FilterTreeWidget(QWidget *parent = 0);
		~FilterTreeWidget();

		void addCamera(View * view);
		void redrawGL();
		void toggle_drrs();

	private:
		void printTree();
		void resetFilterTree();

		QTreeWidgetItem* item_contextMenu; 
		
		//CameraView Actions
		QAction * action_LoadSettings;
		QAction * action_SaveSettings;
		
		//ModelView Actions
		QAction * action_AddSobelFilter;
		QAction * action_AddContrastFilter;
		QAction * action_AddGaussianFilter;
		QAction * action_AddSharpenFilter;

		//Filter Actions
		QAction * action_RemoveFilter;
	protected:
		void drawRow( QPainter* p, const QStyleOptionViewItem &opt, const QModelIndex &idx ) const;
		void dropEvent ( QDropEvent * event );
		void dragMoveEvent(QDragMoveEvent *event);

	public slots:
		void onCustomContextMenuRequested(const QPoint& pos);
		void showContextMenu(QTreeWidgetItem* item, const QPoint& globalPos);

		void action_LoadSettings_triggered();
		void action_SaveSettings_triggered();

		void action_AddSobelFilter_triggered();
		void action_AddContrastFilter_triggered();
		void action_AddGaussianFilter_triggered();
		void action_AddSharpenFilter_triggered();

		void action_RemoveFilter_triggered();
};

#endif  // UAUTOSCOPERMAINWINDOW_H
