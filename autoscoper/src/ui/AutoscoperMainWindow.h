#ifndef AUTOSCOPERMAINWINDOW_H
#define AUTOSCOPERMAINWINDOW_H

#include <QMainWindow>
#include "core/History.hpp"

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
class TrackingOptionsDialog;
class QGLContext;
class Manip3D;
struct GraphData;

class AutoscoperMainWindow : public QMainWindow{

	Q_OBJECT

	public:
		explicit AutoscoperMainWindow(bool skipGpuDevice = false, QWidget *parent = 0);
		~AutoscoperMainWindow();

		Tracker * getTracker(){return tracker;};
		Manip3D * getManipulator(){return manipulator;}
		CoordFrame * getVolume_matrix(){return volume_matrix;}
		void setVolume_matrix(CoordFrame matrix);
		std::vector<unsigned int> *getTextures(){return &textures;}

		GraphData* getPosition_graph();
		void update_graph_min_max(int frame = -1);

		void redrawGL();
		void setFrame(int frame);
		void update_coord_frame();
		void update_xyzypr();

		void push_state();
		void update_xyzypr_and_coord_frame();
		QString get_filename(bool save = true, QString type = "");
		void update_graph_min_max(GraphData* graph, int frame = -1);
		void frame_changed();
		void runBatch(QString batchfile,bool saveData = false);

	private:
		Ui::AutoscoperMainWindow *ui;
		FilterDockWidget* filters_widget;
		TimelineDockWidget* timeline_widget;
		TrackingOptionsDialog* tracking_dialog;

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

		//History
		History *history;
		bool first_undo;
		void undo_state();
		void redo_state();

		//Sets Up all the views;
		void setupUI();
		void openTrial();
		void openTrial(QString filename);
		void newTrial();

		//Shortcuts
		void setupShortcuts();

		//temporary maybe rename/order
		void timelineSetValue(int value);

		void set_manip_matrix(const CoordFrame& frame);
		std::vector<unsigned int> textures;
		void reset_graph();
		
		void save_tracking_prompt();
		void save_trial_prompt();
		void save_tracking_results(QString filename, bool save_as_matrix,bool save_as_rows,bool save_with_commas,bool convert_to_cm,bool convert_to_rad,bool interpolate);
		void save_tracking_results(QString filename);
		void load_tracking_results(QString filename);
		void load_tracking_results(QString filename, bool save_as_matrix,bool save_as_rows,bool save_with_commas,bool convert_to_cm,bool convert_to_rad,bool interpolate);
	protected:
		void closeEvent(QCloseEvent *event);

	public slots:
		
		//File
		void on_actionNew_triggered(bool checked);
		void on_actionOpen_triggered(bool checked);
		void on_actionSave_triggered(bool checked);
		void on_actionSave_as_triggered(bool checked);
		void on_actionImport_Tracking_triggered(bool checked);
		void on_actionExport_Tracking_triggered(bool checked);
		void on_actionQuit_triggered(bool checked);
		void on_actionSaveForBatch_triggered(bool checked);
		void on_actionLoad_xml_batch_triggered(bool checked);

		//Edit
		void on_actionUndo_triggered(bool checked);
		void on_actionRedo_triggered(bool checked);
		void on_actionCut_triggered(bool checked);
		void on_actionCopy_triggered(bool checked);
		void on_actionPaste_triggered(bool checked);
		void on_actionDelete_triggered(bool checked);

		//Tracking
		void on_actionImport_triggered(bool checked);
		void on_actionExport_triggered(bool checked);
		void on_actionInsert_Key_triggered(bool checked);
		void on_actionLock_triggered(bool checked);
		void on_actionUnlock_triggered(bool checked);
		void on_actionBreak_Tangents_triggered(bool checked);
		void on_actionSmooth_Tangents_triggered(bool checked);

		//View
		void on_actionLayoutCameraViews_triggered(bool checked);

		//Toolbar
		void on_toolButtonOpenTrial_clicked();
		void on_toolButtonSaveTracking_clicked();
		void on_toolButtonLoadTracking_clicked();

		void on_toolButtonTranslate_clicked();
		void on_toolButtonRotate_clicked();
		void on_toolButtonMovePivot_clicked();

		void on_toolButtonTrack_clicked();
		void on_toolButtonRetrack_clicked();
		//Shortcuts	
		void key_w_pressed();	
		void key_e_pressed();	
		void key_d_pressed();	
		void key_h_pressed();	
		void key_t_pressed();	
		void key_r_pressed();	
		void key_plus_pressed();
		void key_equal_pressed();
		void key_minus_pressed();

};

#endif  // UAUTOSCOPERMAINWINDOW_H
