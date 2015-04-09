#-------------------------------------------------
#
# Project created by QtCreator 2014-11-06T15:31:43
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = demo_cv
TEMPLATE = app


SOURCES += main.cpp\
        demo_cv.cpp

INCLUDEPATH += /usr/include \
               /usr/include/opencv \
               /usr/include/opencv2

# current opencv version 2.4.8
LIBS += /usr/lib/x86_64-linux-gnu/libopencv_highgui.so \
        /usr/lib/x86_64-linux-gnu/libopencv_core.so \
        /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so

HEADERS  += demo_cv.h

FORMS    += demo_cv.ui
