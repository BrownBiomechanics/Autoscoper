#ifndef TRACKINGOPTIONSDIALOG_H_
#define TRACKINGOPTIONSDIALOG_H_

#include <QDialog>


namespace Ui {
	class TrackingOptionsDialog;
}

class TrackingOptionsDialog : public QDialog{

	Q_OBJECT

	private:
		

	public:
		explicit TrackingOptionsDialog(QWidget *parent = 0);
		~TrackingOptionsDialog();

		Ui::TrackingOptionsDialog *diag;

		
		int frame, from_frame, to_frame, d_frame;
		bool doExit;
		bool frame_optimizing;
		int num_repeats;
		void frame_optimize();
		void setRange(int from, int to, int max);
		void track();
		void retrack();

	public slots:
		void on_pushButton_OK_clicked(bool checked);
		void on_pushButton_Cancel_clicked(bool checked);
		void on_radioButton_CurrentFrame_clicked(bool checked);
		void on_radioButton_PreviousFrame_clicked(bool checked);
		void on_radioButton_LinearExtrapolation_clicked(bool checked);
		void on_radioButton_SplineInterpolation_clicked(bool checked);

};

#endif /* TRACKINGOPTIONSDIALOG_H_ */
