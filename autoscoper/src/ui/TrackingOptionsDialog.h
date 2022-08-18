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

/// \file TrackingOptionsDialog.h
/// \author Benjamin Knorlein, Andy Loomis

#ifndef TRACKINGOPTIONSDIALOG_H_
#define TRACKINGOPTIONSDIALOG_H_

#include <QDialog>


namespace Ui {
  class TrackingOptionsDialog;
}

class TrackingOptionsDialog : public QDialog{

  Q_OBJECT

  private:
    bool is_backup_on;
    int opt_method;
    int cf_model;

  public:
    explicit TrackingOptionsDialog(QWidget *parent = 0);
    ~TrackingOptionsDialog();

    Ui::TrackingOptionsDialog *diag;


    int frame, from_frame, to_frame, d_frame, skip_frame;
    bool doExit;
    bool frame_optimizing;
    int num_repeats;


    // Read random search limits and iterations
    unsigned int max_iter;
    unsigned int max_stall_iter;
    double min_lim;
    double max_lim;


    int curFrame;
    void frame_optimize();
    void setRange(int from, int to, int max);
    //void track();
    //void retrack();
    void trackCurrent();

    bool inActive;

  public slots:
    void on_pushButton_OK_clicked(bool checked);
    void on_pushButton_Cancel_clicked(bool checked);
    void on_radioButton_CurrentFrame_clicked(bool checked);
    void on_radioButton_PreviousFrame_clicked(bool checked);
    void on_radioButton_LinearExtrapolation_clicked(bool checked);
    void on_radioButton_SplineInterpolation_clicked(bool checked);

};

#endif /* TRACKINGOPTIONSDIALOG_H_ */
