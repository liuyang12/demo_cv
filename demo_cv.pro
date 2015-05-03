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

## current opencv version 2.4.8 - ubuntu
#INCLUDEPATH += /usr/include \
#               /usr/include/opencv \
#               /usr/include/opencv2

#LIBS += /usr/lib/x86_64-linux-gnu/libopencv_highgui.so \
#        /usr/lib/x86_64-linux-gnu/libopencv_core.so \
#        /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so

# opencv version 2.4.9 -  windows
INCLUDEPATH += D:\opencv\build\include\opencv\
               D:\opencv\build\include\opencv2\
               D:\opencv\build\include

LIBS += D:\opencv\MinGW\lib\libopencv_calib3d249.dll.a\
        D:\opencv\MinGW\lib\libopencv_contrib249.dll.a\
        D:\opencv\MinGW\lib\libopencv_core249.dll.a\
        D:\opencv\MinGW\lib\libopencv_features2d249.dll.a\
        D:\opencv\MinGW\lib\libopencv_flann249.dll.a\
        D:\opencv\MinGW\lib\libopencv_gpu249.dll.a\
        D:\opencv\MinGW\lib\libopencv_highgui249.dll.a\
        D:\opencv\MinGW\lib\libopencv_imgproc249.dll.a\
        D:\opencv\MinGW\lib\libopencv_legacy249.dll.a\
        D:\opencv\MinGW\lib\libopencv_ml249.dll.a\
        D:\opencv\MinGW\lib\libopencv_objdetect249.dll.a\
        D:\opencv\MinGW\lib\libopencv_video249.dll.a

HEADERS  += demo_cv.h

FORMS    += demo_cv.ui
