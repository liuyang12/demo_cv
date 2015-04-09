#ifndef DEMO_CV_H
#define DEMO_CV_H

#include <QMainWindow>

namespace Ui {
class demo_cv;
}

class demo_cv : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit demo_cv(QWidget *parent = 0);
    ~demo_cv();
    
private:
    Ui::demo_cv *ui;
};

#endif // DEMO_CV_H
