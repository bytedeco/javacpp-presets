include(include.pri)

TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -O2 -D_FORTIFY_SOURCE=2 -fstack-protector-strong -Wformat -Wformat-security
QMAKE_CXXFLAGS += -z noexecstack -z relro -z now
PKGCONFIG += libusb-1.0
CONFIG += link_pkgconfig
INCLUDEPATH += ../include
LIBS += -pthread

LIBS += -L$$DESTDIR/ -lrealsense

PRE_TARGETDEPS += $$DESTDIR/librealsense.a

SOURCES += ../unit-tests/unit-tests-live.cpp \
    ../unit-tests/unit-tests-live-ds-common.cpp \
    ../unit-tests/unit-tests-live-lr200.cpp

HEADERS += ../unit-tests/unit-tests-common.h \
    ../unit-tests/unit-tests-live-ds-common.h

