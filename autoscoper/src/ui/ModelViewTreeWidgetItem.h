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

/// \file ModelViewTreeWidgetItem.h
/// \author Benjamin Knorlein, Andy Loomis

#ifndef MODELVIEWTREEWIDGETITEM_H
#define MODELVIEWTREEWIDGETITEM_H

#include <QTreeWidgetItem>
#include "ui/FilterTreeWidget.h"
#include <QObject>

class CameraTreeWidgetItem;
class FilterTreeWidgetItem;
class FilterTreeWidgetParameter;
class QToolButton;
class QCheckBox;

namespace xromm {
namespace gpu {
class Filter;
}
} // namespace xromm
using xromm::gpu::Filter;

class ModelViewTreeWidgetItem
  : public QObject
  , public QTreeWidgetItem
{
  Q_OBJECT

public:
  ModelViewTreeWidgetItem(int type, std::vector<Filter*>* filters);
  ~ModelViewTreeWidgetItem();

  QString getName() { return name; }
  void setName(QString _name) { name = _name; }
  int getType() { return m_type; }

  std::vector<FilterTreeWidgetParameter*>* getParameters() { return &parameters; }
  void addToCameraTreeWidgetItem(QTreeWidget* treewidget, CameraTreeWidgetItem* cameraWidget);

  void addFilter(FilterTreeWidgetItem* filterItem, bool addToTree = true);
  void removeFilter(FilterTreeWidgetItem* filterItem, bool removeFromTree = true);
  void printFilters();
  void resetVectors();
  void save(std::ofstream& file);
  void loadSettings(std::ifstream& file);
  void loadFilters(std::ifstream& file);
  void toggleVisible();

private:
  void init();
  QString name;

  std::vector<FilterTreeWidgetParameter*> parameters;
  bool settingsShown;

  QFrame* pFrameSettings;
  QToolButton* settingsButton;
  QCheckBox* visibleCheckBox;

  std::vector<FilterTreeWidgetItem*> filterTreeWidgets;
  std::vector<Filter*>* m_filters;
  int m_type;

protected:
public slots:
  void settingsButtonClicked();
  void updateModelview();
  void on_visibleCheckBox_stateChanged(int state);
};

#endif /* MODELVIEWTREEWIDGETITEM_H */
