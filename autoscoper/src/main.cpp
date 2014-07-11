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
	
	if(argc <= 1){ 
		AutoscoperMainWindow *widget = new AutoscoperMainWindow();
		widget->show();
		return app.exec();
	}else{
		fprintf(stderr, "Start Batch %s\n", argv[1]);
		AutoscoperMainWindow *widget = new AutoscoperMainWindow(true);
		widget->runBatch(argv[1], true);
	}
}






