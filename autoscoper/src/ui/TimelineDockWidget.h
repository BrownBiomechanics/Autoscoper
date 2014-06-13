#ifndef TIMELINEDOCKWIDGET_H
#define TIMELINEDOCKWIDGET_H

#include <QDockWidget>
#include "KeyCurve.hpp"

struct GraphData
{
    bool show_x;
    bool show_y;
    bool show_z;
    bool show_yaw;
    bool show_pitch;
    bool show_roll;

    double min_frame;
    double max_frame;
    double min_value;
    double max_value;

    std::vector<bool> frame_locks;
};

enum Selection_type { NODE, IN_TANGENT, OUT_TANGENT };

//forward declarations
namespace Ui {
	class TimelineDockWidget;
}

class QGLContext;
class AutoscoperMainWindow;

class TimelineDockWidget : public QDockWidget{

	Q_OBJECT

	public:
		explicit TimelineDockWidget(QWidget *parent = 0);
		~TimelineDockWidget();
		
		void setSharedGLContext(const QGLContext * sharedContext);
		void setFramesRange(int firstFrame, int lastFrame );
		GraphData* getPosition_graph(){return position_graph;}
		AutoscoperMainWindow * getMainWindow(){return mainwindow;};
		std::vector<std::pair<std::pair<KeyCurve*,KeyCurve::iterator>,Selection_type> > *getSelectedNodes(){return &selected_nodes;}
		void setSelectedNodes(std::vector<std::pair<std::pair<KeyCurve*,KeyCurve::iterator>,Selection_type> > new_nodes){selected_nodes = new_nodes;}

		std::vector<std::pair<KeyCurve*,KeyCurve::iterator> > *getCopiedNodes(){return &copied_nodes;}

		void getValues(double *xyzypr);
		void setValues(double *xyzypr);
		void setValuesEnabled(bool enabled);
		void setFrame(int frame);

		void update_graph_min_max(int frame = -1);

		void setTrial(Trial* trial);
		void draw();
		void setSpinButtonUpdate(bool spinButtonUpdate){m_spinButtonUpdate = spinButtonUpdate;}

	private:
		Ui::TimelineDockWidget *dock;
		GraphData* position_graph;

		AutoscoperMainWindow * mainwindow;
		bool m_spinButtonUpdate;

		std::vector<std::pair<std::pair<KeyCurve*,KeyCurve::iterator>,Selection_type> > selected_nodes;
		std::vector<std::pair<KeyCurve*,KeyCurve::iterator> > copied_nodes;

		//Video play
		int play_tag;
		QTimer *play_timer;

	protected:

	public slots:
		void play_update();

		void on_toolButton_PreviousFrame_clicked();
		void on_toolButton_Stop_clicked();
		void on_toolButton_Play_clicked();
		void on_toolButton_NextFrame_clicked();
		void on_horizontalSlider_Frame_valueChanged(int value);

		void on_doubleSpinBox_X_valueChanged ( double d );
		void on_doubleSpinBox_Y_valueChanged ( double d );
		void on_doubleSpinBox_Z_valueChanged ( double d );
		void on_doubleSpinBox_Yaw_valueChanged ( double d );
		void on_doubleSpinBox_Pitch_valueChanged ( double d );
		void on_doubleSpinBox_Roll_valueChanged ( double d );

		void on_checkBox_X_stateChanged ( int state );
		void on_checkBox_Y_stateChanged ( int state );
		void on_checkBox_Z_stateChanged ( int state );
		void on_checkBox_Yaw_stateChanged ( int state );
		void on_checkBox_Pitch_stateChanged ( int state );
		void on_checkBox_Roll_stateChanged ( int state );

		void on_spinBox_FirstFrame_valueChanged ( int d );
		void on_spinBox_LastFrame_valueChanged ( int d );

};

#endif  // TIMELINEDOCKWIDGET_H
