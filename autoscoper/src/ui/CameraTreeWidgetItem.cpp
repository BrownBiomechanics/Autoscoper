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

/// \file CameraTreeWidgetItem.cpp
/// \author Benjamin Knorlein, Andy Loomis

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui/CameraTreeWidgetItem.h"
#include "ui/ModelViewTreeWidgetItem.h"
#include "ui/FilterTreeWidget.h"

#include <QGridLayout>
#include <QToolButton>
#include <QLabel>
#include <QFileInfo>

#include "View.hpp"
#include "Camera.hpp"

CameraTreeWidgetItem::CameraTreeWidgetItem(View * view):QTreeWidgetItem(CAMERA_VIEW), QObject()
{
  m_view = view;
  init();
}

CameraTreeWidgetItem::CameraTreeWidgetItem(View * view, QTreeWidget * parent):QTreeWidgetItem(parent, CAMERA_VIEW), QObject(parent)
{
  m_view = view;
  init();
}

void CameraTreeWidgetItem::init(){
  QFileInfo fi(m_view->camera()->mayacam().c_str());
  setName(fi.baseName());

  ModelViewTreeWidgetItem* rad = new ModelViewTreeWidgetItem(0, &m_view->radFilters());
  rad->addToCameraTreeWidgetItem(this->treeWidget(),this);

  ModelViewTreeWidgetItem* drr = new ModelViewTreeWidgetItem(1, &m_view->drrFilters());
  drr->addToCameraTreeWidgetItem(this->treeWidget(),this);
}

void CameraTreeWidgetItem::addToGrid(QTreeWidget * treewidget){
  QFrame* pFrame = new QFrame(treewidget);
  pFrame->setMinimumHeight(32);
    QGridLayout* pLayout = new QGridLayout(pFrame);
    pLayout->addWidget(new QLabel(name), 0,0);
  pLayout->setMargin(3);
  treewidget->addTopLevelItem(this);
    treewidget->setItemWidget(this, 0, pFrame);
  setExpanded(true);
  this->setBackgroundColor(0,QColor::fromRgb(230,230,230));
  this->setFlags(this->flags() & ~Qt::ItemIsDragEnabled & ~Qt::ItemIsDropEnabled);
}

CameraTreeWidgetItem::~CameraTreeWidgetItem()
{
}

void CameraTreeWidgetItem::addModelView(ModelViewTreeWidgetItem* modelViewTtem){
  modelViewTreeWidgets.push_back(modelViewTtem);
  this->addChild(modelViewTtem);
}

void CameraTreeWidgetItem::removeModelView(ModelViewTreeWidgetItem* modelViewTtem){
  modelViewTreeWidgets.erase(std::remove(modelViewTreeWidgets.begin(), modelViewTreeWidgets.end(), modelViewTtem), modelViewTreeWidgets.end());
  this->removeChild(modelViewTtem);
}