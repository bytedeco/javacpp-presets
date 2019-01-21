#include <QtCore/QDebug>

#ifdef __APPLE__
#include <pthread.h>
#endif

void QtCore_verifyMainThread() {
#ifdef __APPLE__
    if (!pthread_main_np()) {
        qWarning("\n\n\nWARNING!!\n\n\n"
                 "Qt does not appear to be running on the main thread and will most likely be "
                 "unstable and crash. Please make sure to launch your 'java' command with the "
                 "'-XstartOnFirstThread' command line option. For instance:\n\n"
                 "> java -XstartOnFirstThread org.bytedeco.javacpp.samples.qt.CalculatorExample\n\n");
    }
#endif
}
