/*
 * ProgressDialog.h
 *
 *  Created on: Nov 19, 2013
 *      Author: ben
 */

#ifndef FILTERTREEWIDGETITEM_H
#define FILTERTREEWIDGETITEM_H

#include <QTreeWidgetItem>
#include "ui/FilterTreeWidget.h" 
#include <QObject>

class ModelViewTreeWidgetItem;
class FilterTreeWidgetParameter;
class QToolButton;
class QCheckBox;

namespace xromm{
	namespace gpu{
		class Filter;
	}
}
using xromm::gpu::Filter;

class FilterTreeWidgetItem :  public QObject ,public  QTreeWidgetItem
{
    Q_OBJECT

public:
	FilterTreeWidgetItem(int type);
    //FilterTreeWidgetItem(QString _name);
	//FilterTreeWidgetItem(QString _name, QTreeWidget * parent);
	~FilterTreeWidgetItem();

	QString getName(){return name;}
	void setName(QString _name){name = _name;}

	void addToModelViewTreeWidgetItem(QTreeWidget * treewidget, ModelViewTreeWidgetItem * modelViewWidget, bool addToTree = true);

	Filter * getFilter () {return m_filter;}

private:
	void init();
	QString name;

	std::vector<FilterTreeWidgetParameter * > parameters;
	bool settingsShown;

	QFrame *pFrameSettings;
	QToolButton* settingsButton;
	QCheckBox* visibleCheckBox;

	int m_type;
	Filter * m_filter;

protected:

public slots:
	void settingsButtonClicked();
	void updateFilter();
	void on_visibleCheckBox_stateChanged ( int state );
};



#endif /* FILTERTREEWIDGETITEM_H */
