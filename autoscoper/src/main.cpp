#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <QApplication>

#include "ui/AutoscoperMainWindow.h"
#ifdef _MSC_VER
#        pragma comment(linker, "/SUBSYSTEM:CONSOLE")
#endif


int main ( int argc, char **argv )
{
	QApplication app (argc, argv);

	AutoscoperMainWindow *widget = new AutoscoperMainWindow();
	widget->show();

	return app.exec();
}






