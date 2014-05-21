/*
 * ProgressDialog.h
 *
 *  Created on: Nov 19, 2013
 *      Author: ben
 */

#ifndef CAMERATREEWIDGETITEM_H
#define CAMERATREEWIDGETITEM_H

#include <QTreeWidgetItem>
#include <QObject>

class ModelViewTreeWidgetItem;
namespace xromm{
	namespace gpu{
		class View;
	}
}
using xromm::gpu::View;

class CameraTreeWidgetItem : public QObject ,public  QTreeWidgetItem
{
    Q_OBJECT

public:
    CameraTreeWidgetItem(View * view);
	CameraTreeWidgetItem(View * view, QTreeWidget * parent);
	~CameraTreeWidgetItem();

	QString getName(){return name;}
	void setName(QString _name){name = _name;}

	void addModelView(ModelViewTreeWidgetItem* modelViewWidget);
	void removeModelView(ModelViewTreeWidgetItem* modelViewWidget);

	void addToGrid(QTreeWidget * treewidget);

	View * getView(){return m_view;}

private:
	void init();
	QString name;

	std::vector <ModelViewTreeWidgetItem *> modelViewTreeWidgets;
	View * m_view;

protected:

};



#endif /* CAMERATREEWIDGETITEM_H */
