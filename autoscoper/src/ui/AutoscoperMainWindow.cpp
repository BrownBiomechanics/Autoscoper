#include "ui/AutoscoperMainWindow.h"
#include "ui_AutoscoperMainWindow.h"
#include <QtGui/QGridLayout>
#include "ui/FilterDockWidget.h"
#include "ui/CameraViewWidget.h"
#include "ui/TimelineDockWidget.h"
#include "ui/GLTracker.h"
#include "Manip3D.hpp"

#include "Trial.hpp"
#include "View.hpp"
#include "Tracker.hpp"
#include "CoordFrame.hpp"

#include <QSplitter>
#include <QInputDialog>
#include <QList>
#include <QFileDialog>
#include <QGLContext>

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

AutoscoperMainWindow::AutoscoperMainWindow(QWidget *parent) :
												QMainWindow(parent),
												ui(new Ui::AutoscoperMainWindow){
	
	//Setup UI
	ui->setupUi(this);

	//Init Tracker and get SharedGLContext
	tracker = new Tracker();
	gltracker = new GLTracker(tracker,NULL);
	shared_glcontext = gltracker->context();
	
	//Create Manipulator and
	manipulator = new Manip3D();
	volume_matrix = NULL;

	//Init empty trial
	trial_filename = "";
	is_trial_saved = true;
	is_tracking_saved = true;

	filters_widget =  new FilterDockWidget(this);
	this->addDockWidget(Qt::LeftDockWidgetArea, filters_widget);

	timeline_widget =  new TimelineDockWidget(this);
	this->addDockWidget(Qt::BottomDockWidgetArea, timeline_widget);
	timeline_widget->setSharedGLContext(shared_glcontext);
}

AutoscoperMainWindow::~AutoscoperMainWindow(){
	delete ui;
	delete filters_widget;

	delete tracker;
	delete manipulator;
	if(volume_matrix) delete volume_matrix;

	for (int i = 0 ; i < cameraViews.size();i++){
		delete cameraViews[i];
	}
	cameraViews.clear();
}

void AutoscoperMainWindow::setVolume_matrix(CoordFrame matrix){
	delete volume_matrix;
	volume_matrix = new CoordFrame(matrix);
}

GraphData* AutoscoperMainWindow::getPosition_graph(){
	return timeline_widget->getPosition_graph();
}

void AutoscoperMainWindow::update_graph_min_max(int frame){
	timeline_widget->update_graph_min_max(frame);
}

void AutoscoperMainWindow::on_actionLayoutCameraViews_triggered(bool triggered){
	
	bool ok;
    int rows = QInputDialog::getInteger(this, tr("Layput Camera Views"),
                                          tr("Number of Rows"),1,1,10,1, &ok);
    if (ok)relayoutCameras(rows);
}

void AutoscoperMainWindow::relayoutCameras(int rows){
	//clear Old Splitters
	for (int i = 0 ; i < cameraViews.size();i++){
		cameraViews[i]->setParent(NULL);
	}

	QObjectList objs = ui->frameWindows->children();
	for(int x = 0; x < objs.size(); x ++){
		QObjectList objs2 = objs[x]->children();
		QSplitter * vertsplit = dynamic_cast<QSplitter*> (objs[x]);
		for(int y = objs2.size() - 1 ; y >= 0; y --){	
			QSplitter * horsplit = dynamic_cast<QSplitter*> (objs2[y]);
			if(horsplit){
				QObjectList objs3 = horsplit->children();
				delete horsplit;
			}
		}	
		
		if(vertsplit){
			ui->gridLayout->removeWidget(vertsplit);
			delete vertsplit;
		}
	}
	//Create New
	cameraViewArrangement = QSize(ceil( ((double) cameraViews.size())/rows),rows);

	QSplitter * splitter = new QSplitter(this);
    splitter->setOrientation(Qt::Vertical);
	for(int i = 0; i < cameraViewArrangement.height() ; i ++){
		QSplitter * splitterHorizontal = new QSplitter(splitter);
		splitterHorizontal->setOrientation(Qt::Horizontal);
		splitter->addWidget(splitterHorizontal);
	}

	int freeSpaces = cameraViewArrangement.height() * cameraViewArrangement.width() - cameraViews.size();
	
	int count = 0;
	for (int i = 0 ; i < cameraViews.size();i++, count++){
		if(cameraViews.size() < i + freeSpaces) count++;
		QObject* obj = splitter->children().at(rows - 1 - count / (cameraViewArrangement.width()));
		QSplitter * horsplit = dynamic_cast<QSplitter*> (obj);
		if(horsplit)horsplit->addWidget(cameraViews[i]);
	}

	for(int i = 0; i < cameraViewArrangement.height() ; i ++){
		QObject* obj = splitter->children().at(i);
		QSplitter * horsplit = dynamic_cast<QSplitter*> (obj);
		if(horsplit){
			QList<int> sizelist = horsplit->sizes();
			for(int m = 0; m < sizelist.size(); m++){
					sizelist[m] = 1;
			}
			horsplit->setSizes(sizelist);
		}
	}


	ui->gridLayout->addWidget(splitter, 0, 0, 1, 1);
}

void AutoscoperMainWindow::on_toolButtonOpenTrial_clicked(){
	QString cfg_fileName = QFileDialog::getOpenFileName(this,
									tr("Open Config File"), QDir::currentPath(),tr("Textfiles (*.cfg)"));
	if ( cfg_fileName.isNull() == false )
    {
        try {
			Trial * trial = new Trial(cfg_fileName.toStdString().c_str());
			tracker->load(*trial);
			delete trial;

			trial_filename = cfg_fileName.toStdString();
			is_trial_saved = true;
			is_tracking_saved = true;

			manipulator->set_transform(Mat4d());
			if(volume_matrix) delete volume_matrix;
			volume_matrix = new CoordFrame();

			setupUI();
			timelineSetValue(0);

			timeline_widget->setTrial(tracker->trial());

		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
	}
}

void AutoscoperMainWindow::on_toolButtonTranslate_clicked(){
	ui->toolButtonTranslate->setChecked(true);
	ui->toolButtonRotate->setChecked(false);
	getManipulator()->set_mode(Manip3D::TRANSLATION);
	redrawGL();
}

void AutoscoperMainWindow::on_toolButtonRotate_clicked(){
	ui->toolButtonRotate->setChecked(true);
	ui->toolButtonTranslate->setChecked(false);
	getManipulator()->set_mode(Manip3D::ROTATION);
	redrawGL();
}

void AutoscoperMainWindow::on_toolButtonMovePivot_clicked(){
	getManipulator()->set_movePivot(ui->toolButtonMovePivot->isChecked());
	redrawGL();
}

void AutoscoperMainWindow::timelineSetValue(int value){
	tracker->trial()->frame = value;
    frame_changed();
}

void AutoscoperMainWindow::frame_changed()
{
    // Lock or unlock the position

    /*if (position_graph.frame_locks.at(tracker.trial()->frame)) {
        gtk_widget_set_sensitive(x_spin_button,false);
        gtk_widget_set_sensitive(y_spin_button,false);
        gtk_widget_set_sensitive(z_spin_button,false);
        gtk_widget_set_sensitive(yaw_spin_button,false);
        gtk_widget_set_sensitive(pitch_spin_button,false);
        gtk_widget_set_sensitive(roll_spin_button,false);
    }
    else {
        gtk_widget_set_sensitive(x_spin_button,true);
        gtk_widget_set_sensitive(y_spin_button,true);
        gtk_widget_set_sensitive(z_spin_button,true);
        gtk_widget_set_sensitive(yaw_spin_button,true);
        gtk_widget_set_sensitive(pitch_spin_button,true);
        gtk_widget_set_sensitive(roll_spin_button,true);
    }*/

    update_xyzypr_and_coord_frame();

    for (unsigned int i = 0; i < tracker->trial()->cameras.size(); ++i) {
        tracker->trial()->videos.at(i).set_frame(tracker->trial()->frame);
        tracker->view(i)->radRenderer()->set_rad(
            tracker->trial()->videos.at(i).data(),
            tracker->trial()->videos.at(i).width(),
            tracker->trial()->videos.at(i).height(),
            tracker->trial()->videos.at(i).bps());

        glBindTexture(GL_TEXTURE_2D,textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     1,
                     tracker->trial()->videos.at(i).width(),
                     tracker->trial()->videos.at(i).height(),
                     0,
                     GL_LUMINANCE,
                    (tracker->trial()->videos.at(i).bps() == 8? GL_UNSIGNED_BYTE:
                                                               GL_UNSIGNED_SHORT),
                     tracker->trial()->videos.at(i).data());
        glBindTexture(GL_TEXTURE_2D,0);
    }

	redrawGL();
	QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}


void AutoscoperMainWindow::update_xyzypr()
{
    double xyzypr[6];
    (CoordFrame::from_matrix(trans(manipulator->transform())) * *volume_matrix).to_xyzypr(xyzypr);

    ////Update the spin buttons.
    timeline_widget->setSpinButtonUpdate(false);
    timeline_widget->setValues(&xyzypr[0]);
	timeline_widget->setSpinButtonUpdate(true);
	redrawGL();
}

// Updates the coordinate frames position after the spin buttons values have
// been changed.
void AutoscoperMainWindow::update_xyzypr_and_coord_frame()
{
    if (tracker->trial()->x_curve.empty()) {
        return;
    }

    double xyzypr[6];
    xyzypr[0] = tracker->trial()->x_curve(tracker->trial()->frame);
    xyzypr[1] = tracker->trial()->y_curve(tracker->trial()->frame);
    xyzypr[2] = tracker->trial()->z_curve(tracker->trial()->frame);
    xyzypr[3] = tracker->trial()->yaw_curve(tracker->trial()->frame);
    xyzypr[4] = tracker->trial()->pitch_curve(tracker->trial()->frame);
    xyzypr[5] = tracker->trial()->roll_curve(tracker->trial()->frame);

    CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
    set_manip_matrix(newCoordFrame*volume_matrix->inverse());

	timeline_widget->setSpinButtonUpdate(false);
    timeline_widget->setValues(&xyzypr[0]);
	timeline_widget->setSpinButtonUpdate(true);
}

void AutoscoperMainWindow::set_manip_matrix(const CoordFrame& frame)
{
    double m[16];
    frame.to_matrix_row_order(m);
    manipulator->set_transform(Mat4d(m));
}

void AutoscoperMainWindow::reset_graph()
{
    tracker->trial()->x_curve.clear();
    tracker->trial()->y_curve.clear();
    tracker->trial()->z_curve.clear();
    tracker->trial()->yaw_curve.clear();
    tracker->trial()->pitch_curve.clear();
    tracker->trial()->roll_curve.clear();

    //copied_nodes.clear();
}

// Automatically updates the graph's minimum and maximum values to stretch the
// data the full height of the viewport.
void AutoscoperMainWindow::update_graph_min_max(GraphData* graph, int frame)
{
    if (!tracker->trial() || tracker->trial()->x_curve.empty()) {
        graph->max_value = 180.0;
        graph->min_value = -180.0;
    }
    // If a frame is specified then only check that frame for a new minimum and
    // maximum.
    else if (frame != -1) {
        if (graph->show_x) {
            float x_value = tracker->trial()->x_curve(frame);
            if (x_value > graph->max_value) {
                graph->max_value = x_value;
            }
            if (x_value < graph->min_value) {
                graph->min_value = x_value;
            }
        }
        if (graph->show_y) {
            float y_value = tracker->trial()->y_curve(frame);
            if (y_value > graph->max_value) {
                graph->max_value = y_value;
            }
            if (y_value < graph->min_value) {
                graph->min_value = y_value;
            }
        }
        if (graph->show_z) {
            float z_value = tracker->trial()->z_curve(frame);
            if (z_value > graph->max_value) {
                graph->max_value = z_value;
            }
            if (z_value < graph->min_value) {
                graph->min_value = z_value;
            }
        }
        if (graph->show_yaw) {
            float yaw_value = tracker->trial()->yaw_curve(frame);
            if (yaw_value > graph->max_value) {
                graph->max_value = yaw_value;
            }
            if (yaw_value < graph->min_value) {
                graph->min_value = yaw_value;
            }
        }
        if (graph->show_pitch) {
            float pitch_value = tracker->trial()->pitch_curve(frame);
            if (pitch_value > graph->max_value) {
                graph->max_value = pitch_value;
            }
            if (pitch_value < graph->min_value) {
                graph->min_value = pitch_value;
            }
        }
        if (graph->show_roll) {
            float roll_value = tracker->trial()->roll_curve(frame);
            if (roll_value > graph->max_value) {
                graph->max_value = roll_value;
            }
            if (roll_value < graph->min_value) {
                graph->min_value = roll_value;
            }
        }
    }
    // Otherwise we need to check all the frames.
    else {

        graph->min_value = 1e6;
        graph->max_value = -1e6;

        if (graph->show_x) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float x_value = tracker->trial()->x_curve(frame);
                if (x_value > graph->max_value) {
                    graph->max_value = x_value;
                }
                if (x_value < graph->min_value) {
                    graph->min_value = x_value;
                }
            }
        }
        if (graph->show_y) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float y_value = tracker->trial()->y_curve(frame);
                if (y_value > graph->max_value) {
                    graph->max_value = y_value;
                }
                if (y_value < graph->min_value) {
                    graph->min_value = y_value;
                }
            }
        }
        if (graph->show_z) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float z_value = tracker->trial()->z_curve(frame);
                if (z_value > graph->max_value) {
                    graph->max_value = z_value;
                }
                if (z_value < graph->min_value) {
                    graph->min_value = z_value;
                }
            }
        }
        if (graph->show_yaw) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float yaw_value = tracker->trial()->yaw_curve(frame);
                if (yaw_value > graph->max_value) {
                    graph->max_value = yaw_value;
                }
                if (yaw_value < graph->min_value) {
                    graph->min_value = yaw_value;
                }
            }
        }
        if (graph->show_pitch) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float pitch_value = tracker->trial()->pitch_curve(frame);
                if (pitch_value > graph->max_value) {
                    graph->max_value = pitch_value;
                }
                if (pitch_value < graph->min_value) {
                    graph->min_value = pitch_value;
                }
            }
        }
        if (graph->show_roll) {
            for (frame = floor(graph->min_frame);
                 frame < graph->max_frame;
                 frame += 1.0f) {
                float roll_value = tracker->trial()->roll_curve(frame);
                if (roll_value > graph->max_value) {
                    graph->max_value = roll_value;
                }
                if (roll_value < graph->min_value) {
                    graph->min_value = roll_value;
                }
            }
        }

        graph->min_value -= 1.0;
        graph->max_value += 1.0;
    }
}

void AutoscoperMainWindow::setupUI()
{
    //Remove previous cameras
    for (unsigned int i = 0; i < cameraViews.size(); i++) {
		cameraViews[i]->setParent(NULL);
		delete cameraViews[i];
    }
	cameraViews.clear();
	filters_widget->clearTree();

    //Add the new cameras
    for (unsigned int i = 0; i < tracker->trial()->cameras.size(); i++) {
		cameraViews.push_back(new CameraViewWidget(i, tracker->view(i),tracker->trial()->cameras[i].mayacam().c_str(), this));
		cameraViews[i]->setSharedGLContext(shared_glcontext);	
		filters_widget->addCamera(tracker->view(i));
    }
	relayoutCameras(1);
    textures.resize(tracker->trial()->cameras.size());
    for (unsigned i = 0; i < textures.size(); i++) {
        glGenTextures(1,&textures[i]);
        glBindTexture(GL_TEXTURE_2D,textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     1,
                     tracker->trial()->videos.at(i).width(),
                     tracker->trial()->videos.at(i).height(),
                     0,
                     GL_LUMINANCE,
                    (tracker->trial()->videos.at(i).bps() == 8? GL_UNSIGNED_BYTE:
                                                               GL_UNSIGNED_SHORT),
                     tracker->trial()->videos.at(i).data());
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D,0);
    }
	// Setup the default view

    // Update the number of frames
	timeline_widget->setFramesRange(0,tracker->trial()->num_frames);
    reset_graph();

    // Update the coordinate frames
	timeline_widget->getPosition_graph()->min_frame = 0;
	timeline_widget->getPosition_graph()->max_frame = tracker->trial()->num_frames-1; //
    timeline_widget->getPosition_graph()->frame_locks = vector<bool>(tracker->trial()->num_frames,false);

    update_graph_min_max(timeline_widget->getPosition_graph());

    frame_changed();
}

void AutoscoperMainWindow::redrawGL(){
	for (unsigned int i = 0; i < cameraViews.size(); i++) {
		cameraViews[i]->draw();
    }
	timeline_widget->draw();
}

void AutoscoperMainWindow::setFrame(int frame){
	tracker->trial()->frame = frame;
    frame_changed();
}

void AutoscoperMainWindow::update_coord_frame()
{
    double xyzypr[6];
	timeline_widget->getValues(&xyzypr[0]);
	if(volume_matrix) {
		CoordFrame newCoordFrame = CoordFrame::from_xyzypr(xyzypr);
		CoordFrame mat = newCoordFrame*volume_matrix->inverse();
		set_manip_matrix(newCoordFrame*volume_matrix->inverse());
	}
	redrawGL();
}

void AutoscoperMainWindow::on_actionInsert_Key_triggered(bool checked){
	//push_state();
    //selected_nodes.clear();

    double xyzypr[6];
    (CoordFrame::from_matrix(trans(getManipulator()->transform()))* *getVolume_matrix()).to_xyzypr(xyzypr);

    getTracker()->trial()->x_curve.insert(getTracker()->trial()->frame,xyzypr[0]);
    getTracker()->trial()->y_curve.insert(getTracker()->trial()->frame,xyzypr[1]);
    getTracker()->trial()->z_curve.insert(getTracker()->trial()->frame,xyzypr[2]);
    getTracker()->trial()->yaw_curve.insert(getTracker()->trial()->frame,xyzypr[3]);
    getTracker()->trial()->pitch_curve.insert(getTracker()->trial()->frame,xyzypr[4]);
    getTracker()->trial()->roll_curve.insert(getTracker()->trial()->frame,xyzypr[5]);

	timeline_widget->update_graph_min_max();

	redrawGL();
}