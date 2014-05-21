#ifndef FILTERTREEWIDGETPARAMETER_H
#define FILTERTREEWIDGETPARAMETER_H

#include <QObject>

class FilterTreeWidgetParameter : public QObject{

	Q_OBJECT

	public:
		FilterTreeWidgetParameter(QString _name, double _value, double _minimumValue, double _maximumValue, double _step);
		~FilterTreeWidgetParameter();

		QString name;
		double value;
		double minimumValue;
		double maximumValue;
		double step;
		
	private:
		
	protected:

	public slots:
		void valueChanged(double _value);
		//void action_RemoveFilter_triggered();

	 signals:
		void parameterChanged();

};

#endif  // FILTERTREEWIDGETPARAMETER_H