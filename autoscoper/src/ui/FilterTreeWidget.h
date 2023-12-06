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

/// \file FilterTreeWidget.h
/// \author Benjamin Knorlein, Andy Loomis

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

    void saveAllSettings(QString directory);
    void loadAllSettings(QString directory);
    void loadFilterSettings(int camera, QString filename);

    void clearFilters();
    void setupFilterTuning();
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
