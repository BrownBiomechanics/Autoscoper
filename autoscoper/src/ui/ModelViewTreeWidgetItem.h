/*
 * ProgressDialog.h
 *
 *  Created on: Nov 19, 2013
 *      Author: ben
 */

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

namespace xromm{
	namespace gpu{
		class Filter;
	}
}
using xromm::gpu::Filter;

class ModelViewTreeWidgetItem : public QObject ,public  QTreeWidgetItem
{
    Q_OBJECT

public:
    ModelViewTreeWidgetItem(int type, std::vector<Filter*>* filters);
	
	//ModelViewTreeWidgetItem(QString _name, std::vector<Filter*>* filters);
	//ModelViewTreeWidgetItem(QString _name, std::vector<Filter*>* filters, QTreeWidget * parent);
	~ModelViewTreeWidgetItem();

	QString getName(){return name;}
	void setName(QString _name){name = _name;}

	void addToCameraTreeWidgetItem(QTreeWidget * treewidget, CameraTreeWidgetItem * cameraWidget);

	void addFilter(FilterTreeWidgetItem* filterItem, bool addToTree = true);
	void removeFilter(FilterTreeWidgetItem* filterItem,bool removeFromTree = true);
	void printFilters();
	void resetVectors();

private:
	void init();
	QString name;

	std::vector<FilterTreeWidgetParameter * > parameters;
	bool settingsShown;

	QFrame *pFrameSettings;
	QToolButton* settingsButton;
	QCheckBox* visibleCheckBox;
		
	std::vector <FilterTreeWidgetItem *> filterTreeWidgets;
	std::vector<Filter*>* m_filters;
	int m_type;
protected:

public slots:
	void settingsButtonClicked();
	void updateModelview();
	void on_visibleCheckBox_stateChanged ( int state );
};



#endif /* MODELVIEWTREEWIDGETITEM_H */
