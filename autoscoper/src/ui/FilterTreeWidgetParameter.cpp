#include "ui/FilterTreeWidgetParameter.h"


FilterTreeWidgetParameter::FilterTreeWidgetParameter(QString _name, double _value, double _minimumValue, double _maximumValue, double _step): QObject(),
		name(_name), value(_value) , minimumValue(_minimumValue), maximumValue(_maximumValue), step(_step) 
{
}

FilterTreeWidgetParameter::~FilterTreeWidgetParameter()
{
}

void FilterTreeWidgetParameter::valueChanged(double _value){
	value = _value;
	emit parameterChanged();
}

