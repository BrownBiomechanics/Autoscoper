// ----------------------------------
// Copyright (c) 2019, Brown University
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

/// \file AboutAutoscoper.cpp
/// \author Bardiya Akhbari

#include "ui/AboutAutoscoper.h"
#include "ui_AboutAutoscoper.h"

#include <QTextBrowser>

AboutAutoscoper::AboutAutoscoper(QWidget* parent) :
  QDialog(parent),
  about(new Ui::AboutAutoscoper)
{

  about->setupUi(this);

  // Logo (left side)
  about->LogoLabel->setPixmap(QPixmap(":/logo/autoscoper-256x256.png"));

  // Description (right side)
  QTextBrowser* textBrowser = about->descriptionTextBrowser;
  textBrowser->setFontPointSize(25);
  textBrowser->append("Autoscoper");
  textBrowser->setFontPointSize(11);
  textBrowser->append("");
  QString versionString = QString("v%1.%2").arg("2", "8");
  textBrowser->append(versionString);
  textBrowser->append("");
  textBrowser->moveCursor(QTextCursor::Start, QTextCursor::MoveAnchor);

  // Links (bottom)
  textBrowser = about->linksTextBrowser;
  textBrowser->insertHtml(QString(
                            "<table align=\"center\" border=\"0\" width=\"80%\">"
                            "  <tr>"
                            "    <td align=\"center\"><a href=\"https://autoscoper.readthedocs.io/en/latest/about.html#license\">Licensing Information</a></td>"
                            "    <td align=\"center\"><a href=\"https://autoscoper.readthedocs.io/\">Website</a></td>"
                            "  </tr>"
                            "</table>"));

  connect(about->ButtonBox, SIGNAL(rejected()), this, SLOT(close()));
}

AboutAutoscoper::~AboutAutoscoper()
{
  delete about;
}