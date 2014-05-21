#ifndef AUTOSCOPERMAINWINDOW_H
#define AUTOSCOPERMAINWINDOW_H

#include <QMainWindow>

namespace Ui {
	class AutoscoperMainWindow;
}
namespace xromm {
	class Trial;
	class Tracker;
	class CoordFrame;
}
using xromm::Trial;
using xromm::Tracker;
using xromm::CoordFrame;

class GLTracker;
class FilterDockWidget;
class CameraViewWidget;
class TimelineDockWidget;
class QGLContext;
class Manip3D;

struct GraphData;

class AutoscoperMainWindow : public QMainWindow{

	Q_OBJECT

	public:
		explicit AutoscoperMainWindow(QWidget *parent = 0);
		~AutoscoperMainWindow();

		Tracker * getTracker(){return tracker;};
		Manip3D * getManipulator(){return manipulator;}
		CoordFrame * getVolume_matrix(){return volume_matrix;}
		void setVolume_matrix(CoordFrame matrix);

		GraphData* getPosition_graph();
		void update_graph_min_max(int frame = -1);

		void redrawGL();
		void setFrame(int frame);
		void update_coord_frame();
		void update_xyzypr();

	private:
		Ui::AutoscoperMainWindow *ui;
		FilterDockWidget* filters_widget;
		TimelineDockWidget* timeline_widget;

		std::vector <CameraViewWidget * > cameraViews;
		void relayoutCameras(int rows);
		QSize cameraViewArrangement;
		
		//Trial 
		std::string trial_filename;
		bool is_trial_saved;
		bool is_tracking_saved;

		//Tracker
		Tracker * tracker;
		GLTracker * gltracker;
		const QGLContext* shared_glcontext;

		//Manipulator
		Manip3D * manipulator;
		CoordFrame * volume_matrix;

		//Sets Up all the views;
		void setupUI();

		//temporary maybe rename/order
		void timelineSetValue(int value);
		void frame_changed();
		void update_xyzypr_and_coord_frame();
		void set_manip_matrix(const CoordFrame& frame);
		std::vector<unsigned int> textures;
		void reset_graph();
		void update_graph_min_max(GraphData* graph, int frame = -1);

	protected:

	public slots:
		void on_actionLayoutCameraViews_triggered(bool checked);
		void on_toolButtonOpenTrial_clicked();
		void on_toolButtonTranslate_clicked();
		void on_toolButtonRotate_clicked();
		void on_toolButtonMovePivot_clicked();


		
		void on_actionInsert_Key_triggered(bool checked);

};

#endif  // UAUTOSCOPERMAINWINDOW_H
