#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "ui/FilterTreeWidget.h"
#include "ui/QtCategoryButton.h"

#include "ui/CameraTreeWidgetItem.h"
#include "ui/ModelViewTreeWidgetItem.h"
#include "ui/FilterTreeWidgetItem.h"
#include "ui/FilterDockWidget.h"
#include "ui/AutoscoperMainWindow.h"

#include "View.hpp"
#include <math.h>

#include <QBoxLayout>
#include <QPushButton>
#include <QPainter>
#include <QMenu>
#include <QDropEvent>

#include <iostream>
#include <fstream>
#include <sstream>

FilterTreeWidget::FilterTreeWidget(QWidget *parent) :QTreeWidget(parent){
	setContentsMargins( 0, 0, 0, 0 );

	//Setup the context Menu actions
	setContextMenuPolicy(Qt::CustomContextMenu);
	connect(this,
                SIGNAL(customContextMenuRequested(const QPoint&)),
                SLOT(onCustomContextMenuRequested(const QPoint&)));	

	action_LoadSettings = new QAction(tr("&Load Settings"), this);
	connect(action_LoadSettings, SIGNAL(triggered()), this, SLOT(action_LoadSettings_triggered()));
	action_SaveSettings = new QAction(tr("&Save Settings"), this);
	connect(action_SaveSettings, SIGNAL(triggered()), this, SLOT(action_SaveSettings_triggered()));
	

	action_AddSobelFilter = new QAction(tr("&Add Sobelfilter"), this);
	connect(action_AddSobelFilter, SIGNAL(triggered()), this, SLOT(action_AddSobelFilter_triggered()));
	action_AddContrastFilter = new QAction(tr("&Add Contrastfilter"), this);
	connect(action_AddContrastFilter, SIGNAL(triggered()), this, SLOT(action_AddContrastFilter_triggered()));
	action_AddGaussianFilter = new QAction(tr("&Add Gaussianfilter"), this);
	connect(action_AddGaussianFilter, SIGNAL(triggered()), this, SLOT(action_AddGaussianFilter_triggered()));
	action_AddSharpenFilter = new QAction(tr("&Add Sharpenfilter"), this);
	connect(action_AddSharpenFilter, SIGNAL(triggered()), this, SLOT(action_AddSharpenFilter_triggered()));

	action_RemoveFilter = new QAction(tr("&Remove Filter"), this);
	connect(action_RemoveFilter, SIGNAL(triggered()), this, SLOT(action_RemoveFilter_triggered()));

	//Setup Drag and Drop
	setDragEnabled(true);
	setAcceptDrops(true);
	setDragDropMode(QAbstractItemView::InternalMove);
}

FilterTreeWidget::~FilterTreeWidget(){

}

void FilterTreeWidget::addCamera(View * view){
	CameraTreeWidgetItem* cam = new CameraTreeWidgetItem(view,this);
	cam->addToGrid(this);
}

void FilterTreeWidget::onCustomContextMenuRequested(const QPoint& pos) {

    item_contextMenu = itemAt(pos);
    if (item_contextMenu) {
        // Note: We must map the point to global from the viewport to
        // account for the header.
        showContextMenu(item_contextMenu, viewport()->mapToGlobal(pos));
    }
} 
void FilterTreeWidget::showContextMenu(QTreeWidgetItem* item_contextMenu, const QPoint& globalPos) {
    QMenu menu;
	QMenu menuFilters;
    switch (item_contextMenu->type()) {
		case CAMERA_VIEW:
            menu.addAction(action_LoadSettings);
			menu.addAction(action_SaveSettings);
			break;

        case MODEL_VIEW:		
			menuFilters.setTitle("Add Filters");
			menuFilters.addAction(action_AddSobelFilter);
            menuFilters.addAction(action_AddContrastFilter);
			menuFilters.addAction(action_AddGaussianFilter);
            menuFilters.addAction(action_AddSharpenFilter);
			menu.addMenu(&menuFilters);
			break;
 
        case FILTER:
            menu.addAction(action_RemoveFilter);
            break;
    }
 
    menu.exec(globalPos);
}

void FilterTreeWidget::action_LoadSettings_triggered(){
	CameraTreeWidgetItem * cameraTreeItem = dynamic_cast<CameraTreeWidgetItem*> (item_contextMenu); 
	FilterDockWidget * dock_widget = dynamic_cast <FilterDockWidget *>(parent()->parent());

	if(cameraTreeItem && dock_widget){
		QString filename = dock_widget->getMainWindow()->get_filename(false, "*.vie");
		if (filename.compare("") == 0) {
			return;
		}

		std::ifstream file(filename.toStdString().c_str(), std::ios::in);
		if (!file) {
			std::cerr << "Import: Unable to open file for writing" << std::endl;
			return;
		}

		std::string line, key;
		while (std::getline(file,line)) {
			if (line.compare("DrrRenderer_begin") == 0) {
				for(int i = 0 ; i < cameraTreeItem->childCount(); i ++){
					ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (cameraTreeItem->child(i));
					if(modelviewItem && modelviewItem->getType() == 1){
						modelviewItem->loadSettings(file);
					}
				}
			}else if(line.compare("DrrFilters_begin") == 0){
				for(int i = 0 ; i < cameraTreeItem->childCount(); i ++){
					ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (cameraTreeItem->child(i));
					if(modelviewItem && modelviewItem->getType() == 1){
						modelviewItem->loadFilters(file);
					}
				}
			}else if(line.compare("RadFilters_begin") == 0){
				for(int i = 0 ; i < cameraTreeItem->childCount(); i ++){
					ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (cameraTreeItem->child(i));
					if(modelviewItem && modelviewItem->getType() == 0){
						modelviewItem->loadFilters(file);
					}
				}
			}
		}
	}
}
void FilterTreeWidget::action_SaveSettings_triggered(){
	CameraTreeWidgetItem * cameraTreeItem = dynamic_cast<CameraTreeWidgetItem*> (item_contextMenu); 
	FilterDockWidget * dock_widget = dynamic_cast <FilterDockWidget *>(parent()->parent());

	if(cameraTreeItem && dock_widget){
		QString filename = dock_widget->getMainWindow()->get_filename(true, "*.vie");
		if (filename.compare("") == 0) {
			return;
		}

		std::ofstream file(filename.toStdString().c_str(), std::ios::out);
		if (!file) {
			std::cerr << "Export: Unable to open file for writing" << std::endl;
			return;
		}

		for(int i = 0 ; i < cameraTreeItem->childCount(); i ++){
			ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (cameraTreeItem->child(i));
			if(modelviewItem){
				modelviewItem->save(file);
			}
		}

		file.close();
	}
}

void FilterTreeWidget::saveAllSettings(QString directory){
	for(int i=0;i<this->topLevelItemCount(); ++i){
		CameraTreeWidgetItem * camera = dynamic_cast<CameraTreeWidgetItem*> (topLevelItem(i));
		if(camera){
			QString filename = directory + camera->getName() + ".vie"; 
			std::ofstream file(filename.toAscii().constData(), std::ios::out);
			for(int j=0;j<camera->childCount(); ++j){
				ModelViewTreeWidgetItem * model = dynamic_cast<ModelViewTreeWidgetItem*> (camera->child(j));
				if(model){
					model->save(file);
				}
			}
			file.close();
		}
	}
} 

void FilterTreeWidget::loadAllSettings(QString directory){
	for(int i=0;i<this->topLevelItemCount(); ++i){
		CameraTreeWidgetItem * camera = dynamic_cast<CameraTreeWidgetItem*> (topLevelItem(i));
		if(camera){
			QString filename = directory + camera->getName() + ".vie"; 
			std::ifstream file(filename.toStdString().c_str(), std::ios::in);
			std::string line, key;
			while (std::getline(file,line)) {
				if (line.compare("DrrRenderer_begin") == 0) {
					for(int i = 0 ; i < camera->childCount(); i ++){
						ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (camera->child(i));
						if(modelviewItem && modelviewItem->getType() == 1){
							modelviewItem->loadSettings(file);
						}
					}
				}else if(line.compare("DrrFilters_begin") == 0){
					for(int i = 0 ; i < camera->childCount(); i ++){
						ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (camera->child(i));
						if(modelviewItem && modelviewItem->getType() == 1){
							modelviewItem->loadFilters(file);
						}
					}
				}else if(line.compare("RadFilters_begin") == 0){
					for(int i = 0 ; i < camera->childCount(); i ++){
						ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (camera->child(i));
						if(modelviewItem && modelviewItem->getType() == 0){
							modelviewItem->loadFilters(file);
						}
					}
				}
			}
			file.close();
		}
	}
} 

void FilterTreeWidget::toggle_drrs(){
	for(int i=0;i<this->topLevelItemCount(); ++i){
		CameraTreeWidgetItem * camera = dynamic_cast<CameraTreeWidgetItem*> (topLevelItem(i));
		if(camera){
			for(int j=0;j<camera->childCount(); ++j){
				ModelViewTreeWidgetItem * model = dynamic_cast<ModelViewTreeWidgetItem*> (camera->child(j));
				if(model && model->getType() == 1){
					model->toggleVisible();
				}
			}
		}
	}

}

void FilterTreeWidget::action_AddSobelFilter_triggered(){
	ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (item_contextMenu); 
	if(modelviewItem){
		FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(0);
		filter->addToModelViewTreeWidgetItem(this,modelviewItem);
	}
	redrawGL();
}
void FilterTreeWidget::action_AddContrastFilter_triggered(){
	ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (item_contextMenu); 
	if(modelviewItem){
		FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(1);
		filter->addToModelViewTreeWidgetItem(this,modelviewItem);
	}
	redrawGL();
}
void FilterTreeWidget::action_AddGaussianFilter_triggered(){
	ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (item_contextMenu); 
	if(modelviewItem){
		FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(2);
		filter->addToModelViewTreeWidgetItem(this,modelviewItem);
	}
	redrawGL();
}
void FilterTreeWidget::action_AddSharpenFilter_triggered(){
	ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (item_contextMenu); 
	if(modelviewItem){
		FilterTreeWidgetItem* filter = new FilterTreeWidgetItem(3);
		filter->addToModelViewTreeWidgetItem(this,modelviewItem);
	}
	redrawGL();
}
void FilterTreeWidget::action_RemoveFilter_triggered(){
	FilterTreeWidgetItem * filterItem = dynamic_cast<FilterTreeWidgetItem*> (item_contextMenu); 
	ModelViewTreeWidgetItem * modelviewItem = dynamic_cast<ModelViewTreeWidgetItem*> (item_contextMenu->parent()); 
	if(filterItem && modelviewItem){
		modelviewItem->removeFilter(filterItem);
		delete filterItem;	
	}
	redrawGL();
}

void FilterTreeWidget::dragMoveEvent(QDragMoveEvent *event)
 {
	 QTreeWidgetItem * dropped = itemAt( event->pos() );
	 QRect r = visualItemRect(dropped); 

	 if( r.x() + 2  > event->pos().x() || r.y() + 2 > event->pos().y() 
		 || r.x() + r.width() - 3 < event->pos().x() || r.y() + r.height() - 3 < event->pos().y() ){
		event->ignore();
	 }
	 else if (dropped && dropped->type()==MODEL_VIEW ){
		 event->acceptProposedAction();
	 }else if (dropped && dropped->type()==FILTER){
		 event->acceptProposedAction();
	 }else{
		 event->ignore();
	 }
 }
void FilterTreeWidget::dropEvent ( QDropEvent * event ){
	QTreeWidgetItem * dropped = itemAt( event->pos() );
	QTreeWidgetItem * dragged = currentItem();
	ModelViewTreeWidgetItem * modelviewItemDragged = NULL;
	ModelViewTreeWidgetItem * modelviewItemDropped = NULL;

	FilterTreeWidgetItem* draggedFilter = dynamic_cast<FilterTreeWidgetItem*> (dragged);
	QTreeWidget::dropEvent(event);

	if(dropped->type() == MODEL_VIEW) modelviewItemDropped = dynamic_cast<ModelViewTreeWidgetItem*> (dropped); 
	if(dropped->type() == FILTER && dropped->parent()) modelviewItemDropped = dynamic_cast<ModelViewTreeWidgetItem*> (dropped->parent()); 

	if(draggedFilter && modelviewItemDropped){
		draggedFilter->addToModelViewTreeWidgetItem(this,modelviewItemDropped,false);
	}

	resetFilterTree();

	redrawGL();

	//printTree();
}

void FilterTreeWidget::resetFilterTree(){
	for(int i=0;i<this->topLevelItemCount(); ++i){
		CameraTreeWidgetItem * camera = dynamic_cast<CameraTreeWidgetItem*> (topLevelItem(i));
		if(camera){
			for(int j=0;j<camera->childCount(); ++j){
				ModelViewTreeWidgetItem * model = dynamic_cast<ModelViewTreeWidgetItem*> (camera->child(j));
				if(model){
					model->resetVectors();
				}
			}
		}
	}
}
void FilterTreeWidget::printTree(){
	fprintf(stderr,"\n");
	for(int i=0;i<this->topLevelItemCount(); ++i){
		CameraTreeWidgetItem * camera = dynamic_cast<CameraTreeWidgetItem*> (topLevelItem(i));
		if(camera){
			fprintf(stderr,"Camera%d %s\n",i, camera->getName().toStdString().c_str());
			for(int j=0;j<camera->childCount(); ++j){
				ModelViewTreeWidgetItem * model = dynamic_cast<ModelViewTreeWidgetItem*> (camera->child(j));
				if(model){
					fprintf(stderr,"    Model%d %s\n",j,model->getName().toStdString().c_str());
					fprintf(stderr,"---------------------\n");
					model->printFilters();
					fprintf(stderr,"---------------------\n");
					for(int k=0;k<model->childCount(); ++k){
						FilterTreeWidgetItem * filter = dynamic_cast<FilterTreeWidgetItem*> (model->child(k));
						if(filter){
							
							fprintf(stderr,"          Filter%d %s\n",k,filter->getName().toStdString().c_str());
							if(filter->childCount()>0){
								fprintf(stderr,"Filter has Children\n");
							}
						}else{
							fprintf(stderr,"2nd Level no Filter\n");
						}
					}
				}else{
					fprintf(stderr,"1st Level no Model\n");
				}
			}
		}else{
			fprintf(stderr,"Top Level no Camera\n");
		}
	}
}

void FilterTreeWidget::drawRow( QPainter* p, const QStyleOptionViewItem &opt, const QModelIndex &idx ) const
{
    
	QTreeWidget::drawRow( p, opt, idx );
    QModelIndex s = idx.sibling( idx.row(), 0 );
    if ( s.isValid() )
    {
		QRect rect = visualRect( s );
        int py = rect.y();
        int ph = rect.height();
        int pw = rect.width();
		int px = rect.x();
		if(itemFromIndex(s)->type() == CAMERA_VIEW){
			p->setPen( QColor( 0, 0, 0 ) );
			p->drawLine( px, py + ph -1, pw + px, py + ph -1);
			p->drawLine( px, py , pw + px, py );
		}else if(itemFromIndex(s)->type() == MODEL_VIEW){
			p->setPen( QColor( 150, 150, 150 ) );
			p->drawLine( px, py + ph -1, pw + px, py + ph -1);
			p->drawLine( px, py , pw + px, py );
		}else if(itemFromIndex(s)->type() == FILTER){
			p->setPen( QColor( 200, 200, 200 ) );
			p->drawLine( px, py + ph -1, pw + px, py + ph -1);
		}
    }
}

void FilterTreeWidget::redrawGL(){
	FilterDockWidget * dock_widget = dynamic_cast <FilterDockWidget *>(parent()->parent());
	if(dock_widget) dock_widget->getMainWindow()->redrawGL();
}