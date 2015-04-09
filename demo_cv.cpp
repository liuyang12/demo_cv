#include "demo_cv.h"
#include "ui_demo_cv.h"

demo_cv::demo_cv(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::demo_cv)
{
    ui->setupUi(this);
}

demo_cv::~demo_cv()
{
    delete ui;
}
