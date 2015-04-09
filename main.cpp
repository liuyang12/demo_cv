#include "demo_cv.h"
#include <QApplication>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <QDebug>
using namespace std;
using namespace cv;

#define WIDTH 480
#define HEIGHT 360
#define COLOR_Orange  1*32
#define	COLOR_Yellow  2*32
#define	COLOR_Blue  3*32
#define	COLOR_Green 4*32
#define	COLOR_White  5*32
#define	COLOR_Black  0

#define erosion_type MORPH_RECT     // 形态学矩形
#define erosion_size 2              // kernel = 3*3

#define START_NUM 00
#define END_NUM 38     /* 根据读入的文件名进行修改/home/young/文档/test_pics/ball/level1/clear/frame*/

struct kbline{  // line in Slope–intercept form
  double k;
  double b;
};

/// 全局变量
Mat src, erosion_dst, dilation_dst;

int erosion_elem = 0;
//int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

/** Function Headers */
void Erosion( int, void* );
void Dilation( int, void* );

void thinning(const cv::Mat& src, cv::Mat& dst);
void findgate(int i);

int main(int argc, char *argv[])
{

//    void colorReduce(Mat image, int div = 64);
    QApplication a(argc, argv);
//    demo_cv w;
//    Mat src, src_gray, src_op;
    /// Read the image
//    src = imread("/home/young/文档/test_pics/ball/level3/clear/frame0240.jpg", 1 );
    int i = START_NUM;
    for(i = 2; ;)
    {
        if(i > END_NUM)
            i = START_NUM;
        if(i < START_NUM)
            i = END_NUM;
        findgate(i);
        int c = waitKey();
        if(c == 'a' || c == 'A' || c == 's' || c == 'S' || c == 65361 || c == 65364)
            i--;
        else if(c == 'd' || c == 'D' || c == 'w' || c == 'W' || c == 65362 || c == 65363)
            i++;
        else if(c == 10 || c == 27 || c == 32)   // 回车/ESC退出
            break;
        else{
            qDebug() << c;
        }

    }


//    /** 图像腐蚀和膨胀 */
//    /// 创建显示窗口
//    namedWindow( "Erosion Demo", CV_WINDOW_AUTOSIZE );
//    namedWindow( "Dilation Demo", CV_WINDOW_AUTOSIZE );
//    cvMoveWindow( "Dilation Demo", src.cols, 0 );

//    /// 创建腐蚀 Trackbar
//    createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
//                    &erosion_elem, max_elem,
//                    Erosion );

//    createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
//                    &erosion_size, max_kernel_size,
//                    Erosion );

//    /// 创建膨胀 Trackbar
//    createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
//                    &dilation_elem, max_elem,
//                    Dilation );

//    createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
//                    &dilation_size, max_kernel_size,
//                    Dilation );

//    /// Default start
//    Erosion( 0, 0 );
//    Dilation( 0, 0 );
//    /*  */
//    waitKey(0);
    return 0;
//    return a.exec();
}

void findgate(int i)
{
    char file_name[256], source_name[127], thinned_name[127], txt[127];
//    if(i > END_NUM)
//        i = END_NUM;
    std::sprintf(file_name, "/home/young/文档/test_pics/bmp.for_as/bmp_for_all/color_classified/frame%04d.bmp", i);
    qDebug() << file_name;
    src = imread(file_name, 1);
    if( !src.data ){
//        return -1;
        return ;
    }
    qDebug() << src.rows << "*" << src.cols;
    cv::Mat bw, draw;
    cv::cvtColor(src, bw, CV_BGR2GRAY);
    cv::threshold(bw, bw, COLOR_Green+2, 255, CV_THRESH_BINARY);
    std::sprintf(source_name, "source image - frame%04d.bmp",i);
    std::sprintf(thinned_name, "thinned image - frame%04d.bmp",i);

    Mat element = getStructuringElement( erosion_type,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );
    /// 腐蚀操作
    erode(bw, bw, element);  // 首先做一步腐蚀 Kernel = 3*3
    thinning(bw, bw);   // 再进行一步细化
    cv::cvtColor(bw, draw, CV_GRAY2BGR);
    /* 概率的霍夫变换模型 */
    vector<Vec4i> lines;
    //参数解释 - 概率的霍夫变换模型
    //1代表ρ的步进方式为1个像素，CV_PI/180代表θ的步进方式为1度，20为最低阈值即20个点以上构成的直线才认为有效
    //50代表将要返回的线段的最小长度，10代表能连成1条直线的线段最多能分隔的像素点数
    //后两个系数是前两个系数的再细化
    HoughLinesP(bw, lines, 1, CV_PI/180, 20, 20, 10);
    qDebug() << lines.size();
    /* 初始化字体 */
    CvFont font;
    double hScale=1;
    double vScale=1;
    int lineWidth=2;// 相当于写字的线条
//    scalar=CV_RGB(255,0,0);
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);//初始化字体，准备写到图片上的
    /**/
    vector<Vec4i >::iterator itc = lines.begin();
    int t = 0;
    kbline KbLine[50];
    while(itc != lines.end())
    {
        Vec4i l = *itc;
        Point p1 = Point(l[0],l[1]), p2 = Point(l[2],l[3]);
        if((p1.x==p2.x&&p1.x<=WIDTH&&p1.x>=WIDTH-3)||(p1.y==p2.y&&p1.y<=3)||(p1.x==p2.x&&p1.x<=3)||(p1.y==p2.y&&p1.y<=HEIGHT&&p1.y>=HEIGHT-3)){
//            lines.erase(i);
            itc = lines.erase(itc);
            continue;
        }
        KbLine[t].k = 1.0*(p2.y-p1.y)/(p2.x-p1.x);
        KbLine[t].b = 1.0*(p1.y*p2.x-p2.y*p1.x)/(p2.x-p1.x);
        qDebug()<<t<<"("<<(*itc)[0]<<","<<(*itc)[1]<<")->("<<(*itc)[2]<<","<<(*itc)[3]<<")";
        qDebug() << "  (k,b)=(" <<KbLine[t].k << "," << KbLine[t].b << ")";
        line(src, p1, p2, Scalar(0,0,255), 3);
        line(draw, p1, p2, Scalar(0,0,255),1);
        std::sprintf(txt, "%d", t);
        cv::putText(draw, txt, p1, CV_FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0));
//        cv::cvPutText(draw, txt, p1, &font, Scalar(255,0,0));
//        line(draw, Point(0,0), Point(100,300), Scalar(255,0,0), 3);
        ++itc;t++;
    }
    for(size_t t=0; t<lines.size(); t++){
        Vec4i l = lines[t];


    }
    /***/
    namedWindow("source image", CV_WINDOW_AUTOSIZE);
    namedWindow("thinned image", CV_WINDOW_AUTOSIZE);
    cv::imshow("source image", src);
    cv::imshow("thinned image", draw);
//    waitKey();
}

/**  @function Erosion  */
void Erosion( int, void* )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  /// 腐蚀操作
  erode( src, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
}   /* 图像腐蚀 */

/** @function Dilation */
void Dilation( int, void* )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// 膨胀操作
  dilate( src, dilation_dst, element );
  imshow( "Dilation Demo", dilation_dst );
}   /* 图像膨胀 */

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * Parameters:
 * 		im    Binary image with range = [0,1]
 * 		iter  0=even, 1=odd
 */
void thinningIteration(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(y+1);

        pDst = marker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                     (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                     (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                     (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }

    img &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 * 		src  The source image, binary with range = [0,255]
 * 		dst  The destination image
 */
void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();
    dst /= 255;         // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}



//    Mat img(500, 500, CV_8UC3);
//    RNG& rng = theRNG();
//    cout <<  "\n这个程序演示了凸包函数的使用，任意给定一些点，求出包围这些点的凸包\n";
//    for( ; ; )
//    {
//        char key;
//        int i, count = (unsigned)rng%100 + 1;

//        vector<Point> points;
//        //随机在1-100个点，这些点位于图像中心3/4处。
//        for( i = 0; i < count; i++ )
//        {
//            Point pt;
//            pt.x = rng.uniform(img.cols/4, img.cols*3/4);
//            pt.y = rng.uniform(img.rows/4, img.rows*3/4);

//            points.push_back(pt);
//        }
//        //计算凸包
//        vector<int> hull;
//        convexHull(Mat(points), hull, true);

//       //画随即点
//        img = Scalar::all(0);
//        for( i = 0; i < count; i++ )
//            circle(img, points[i], 3, Scalar(0, 0, 255), CV_FILLED, CV_AA);

//        int hullcount = (int)hull.size();
//        Point pt0 = points[hull[hullcount-1]];
//        //画凸包
//        for( i = 0; i < hullcount; i++ )
//        {
//            Point pt = points[hull[i]];
//            line(img, pt0, pt, Scalar(0, 255, 0), 1, CV_AA);
//            pt0 = pt;
//        }

//        imshow("hull", img);

//        key = (char)waitKey();
//        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
//            break;
//    }

//    return a.exec();
//void colorReduce(Mat image, int div = 64)
//{
//    int nrow = image.rows;
//    int ncol = image.cols * image.channels();
//    for(int j = 0; j < nrow; j++)
//    {
//        uchar* data = image.ptr<uchar>(j);
//        for(int i = 0; i < ncol; i++)
//        {
//            data[i] = data[i] / div * div + div / 2;
//        }
//    }
//}

