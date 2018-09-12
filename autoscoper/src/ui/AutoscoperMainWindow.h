// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

/// \file AutoscoperMainWindow.h
/// \author Benjamin Knorlein, Andy Loomis

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
class VolumeDockWidget;
class TrackingOptionsDialog;
class WorldViewWindow;
class QOpenGLContext;
class Manip3D;
struct GraphData;

class AutoscoperMainWindow : public QMainWindow{

	Q_OBJECT

	public:
		explicit AutoscoperMainWindow(bool skipGpuDevice = false, QWidget *parent = 0);
		~AutoscoperMainWindow();

		Tracker * getTracker(){return tracker;};
		Manip3D * getManipulator(int idx = -1);
		std::vector<unsigned int> *getTextures(){return &textures;}

		GraphData* getPosition_graph();
		void update_graph_min_max(int frame = -1);

		void redrawGL();
		void setFrame(int frame);
		void update_coord_frame();
		void update_xyzypr();

		// Getting current frame
		int getCurrentFrame();
		// Storing last used directory
		QString getLastFolder();
		// Current Frame variable
		int curFrame;

		void push_state();
		void update_xyzypr_and_coord_frame();
		QString get_filename(bool save = true, QString type = "");
		void update_graph_min_max(GraphData* graph, int frame = -1);
		void frame_changed();
		void volume_changed();
		void runBatch(QString batchfile,bool saveData = false);

		//For socket
		void openTrial(QString filename);
		void load_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume = -1);
		void save_tracking_results(QString filename, bool save_as_matrix, bool save_as_rows, bool save_with_commas, bool convert_to_cm, bool convert_to_rad, bool interpolate, int volume = -1);
		void loadFilterSettings(int camera, QString filename);
		std::vector<double> getPose(unsigned int volume, unsigned int frame);
		void setPose(std::vector<double> pose, unsigned int volume, unsigned int frame);
		void setBackground(double threshold);
		std::vector <double> getNCC(unsigned int volumeID, double* xyzpr);
		std::vector <unsigned char> getImageData(unsigned int volumeID, unsigned int camera, double* xyzpr, unsigned int &width, unsigned int &height);
		void optimizeFrame(int volumeID, int frame, int dframe, int repeats, int opt_method, unsigned int max_iter, double min_limit, double max_limit, int cf_model, unsigned int stall_iter);


		// Backup Save
		void backup_tracking(bool backup_on);


	private:
		Ui::AutoscoperMainWindow *ui;
		FilterDockWidget* filters_widget;
		TimelineDockWidget* timeline_widget;
		VolumeDockWidget* volumes_widget;
		TrackingOptionsDialog* tracking_dialog;

		std::vector <CameraViewWidget * > cameraViews;
		void relayoutCameras(int rows);
		QSize cameraViewArrangement;
		
		//Trial 
		std::string trial_filename;
		bool is_trial_saved;
		bool is_tracking_saved;

		// Backup
		bool backup_on;

		//Tracker
		Tracker * tracker;
		GLTracker * gltracker;
		QOpenGLContext* shared_glcontext;

		//Manipulator
		std::vector <Manip3D *> manipulator;
		//WortldView
		WorldViewWindow * worldview;

		//Threshold
		double background_threshold_;

		//History
		History *history;
		bool first_undo;
		void undo_state();
		void redo_state();

		//Sets Up all the views;
		void setupUI();
		void openTrial();
		void newTrial();

		//Shortcuts
		void setupShortcuts();

		// For storing the last opened folder
		void setLastFolder(QString lastFolder);
		QString lastFolderPath;

		//temporary maybe rename/order
		void timelineSetValue(int value);

		void set_manip_matrix(int idx, const CoordFrame& frame);
		std::vector<unsigned int> textures;
		void reset_graph();
		
		void save_tracking_prompt();
		void save_trial_prompt();
		void save_tracking_results(QString filename);

		void save_ncc_results(QString filename);
		void save_nearby_nccs(QString filename);

		double rand_gen_main(double fMin, double fMax);

		void load_tracking_results(QString filename);
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
		void on_actionSave_Test_Sequence_triggered(bool checked);
		void on_actionSaveForBatch_triggered(bool checked);
		void on_actionLoad_xml_batch_triggered(bool checked);

		//Edit
		void on_actionUndo_triggered(bool checked);
		void on_actionRedo_triggered(bool checked);
		void on_actionCut_triggered(bool checked);
		void on_actionCopy_triggered(bool checked);
		void on_actionPaste_triggered(bool checked);
		void on_actionDelete_triggered(bool checked);
		void on_actionSet_Background_triggered(bool checked);

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
		void on_actionShow_world_view_triggered(bool checked);


		// Extra
		void on_actionExport_NCC_as_csv_triggered(bool checked);
		void on_actionExport_all_NCCs_near_this_pose_triggered(bool checked);

		//Toolbar
		void on_toolButtonOpenTrial_clicked();
		void on_toolButtonSaveTracking_clicked();
		void on_toolButtonLoadTracking_clicked();

		void on_toolButtonTranslate_clicked();
		void on_toolButtonRotate_clicked();
		void on_toolButtonMovePivot_clicked();

		void on_toolButtonTrack_clicked();
		void on_toolButtonTrackCurrent_clicked();
		void on_toolButtonRetrack_clicked();


		//Shortcuts	
		void key_w_pressed();	
		void key_e_pressed();	
		void key_d_pressed();	
		void key_h_pressed();	
		void key_t_pressed();	
		//void key_r_pressed();	
		void key_p_pressed();
		void key_c_pressed();
		void key_plus_pressed();
		void key_equal_pressed();
		void key_minus_pressed();

};

#endif  // UAUTOSCOPERMAINWINDOW_H
