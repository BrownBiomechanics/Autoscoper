#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui_FilterDockWidget.h"
#include "ui/FilterDockWidget.h"
#include "ui/AutoscoperMainWindow.h"

FilterDockWidget::FilterDockWidget(QWidget *parent) :
										QDockWidget(parent),
										dock(new Ui::FilterDockWidget){
	dock->setupUi(this);

	mainwindow  = dynamic_cast <AutoscoperMainWindow *> ( parent);
}

FilterDockWidget::~FilterDockWidget(){
	delete dock;
}

void FilterDockWidget::clearTree(){

	dock->treeWidget->clear();
}
void FilterDockWidget::toggle_drrs(){
	dock->treeWidget->toggle_drrs();
}

void FilterDockWidget::addCamera(View * view){
	dock->treeWidget->addCamera(view);
}
