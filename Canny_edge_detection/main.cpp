//
// Created by liangdaxin on 23-6-20.
//
#include "util.h"
#include <iostream>
#include <string>
#define SRC_PATH std::string("../src")
#define OUT_PATH std::string("../out")

// 测试Sobel
void test_Sobel(){
    cv::Mat img = cv::imread( SRC_PATH+"/img1.jpg",0);
    cv::Mat x_derivative,y_derivative;
    util::Sobel(img,x_derivative,y_derivative);
    cv::Mat magnitude,direction;
    util::get_gradient(x_derivative,y_derivative,magnitude,direction);
    cv::namedWindow("x_derivative",cv::WINDOW_NORMAL);
    cv::namedWindow("y_derivative",cv::WINDOW_NORMAL);
    cv::namedWindow("magnitude",cv::WINDOW_NORMAL);
    cv::namedWindow("direction",cv::WINDOW_NORMAL);
    cv::resizeWindow("x_derivative",img.size()/3);
    cv::resizeWindow("y_derivative",img.size()/3);
    cv::resizeWindow("magnitude",img.size()/3);
    cv::resizeWindow("direction",img.size()/3);
    cv::imwrite(OUT_PATH+"/x_derivative.png",x_derivative);
    cv::imwrite(OUT_PATH+"/y_derivative.png",y_derivative);
    cv::imwrite(OUT_PATH+"/magnitude.png",magnitude);
    cv::imwrite(OUT_PATH+"/direction.png",direction,{cv::IMWRITE_PNG_COMPRESSION,9});
    cv::imshow("x_derivative",x_derivative);
    cv::imshow("y_derivative",y_derivative);
    cv::imshow("magnitude",magnitude);
    cv::imshow("direction",direction);
    cv::waitKey();
}
// 测试Prewitt
void test_prewitt(){

    cv::Mat img = cv::imread(SRC_PATH+"/img1.jpg",0);
    cv::Mat x_derivative,y_derivative;
    util::Prewitt(img,x_derivative,y_derivative);
    cv::namedWindow("x_derivative",cv::WINDOW_NORMAL);
    cv::namedWindow("y_derivative",cv::WINDOW_NORMAL);
    cv::resizeWindow("x_derivative",img.size()/3);
    cv::resizeWindow("y_derivative",img.size()/3);
    cv::imshow("x_derivative",x_derivative);
    cv::imshow("y_derivative",y_derivative);
    cv::waitKey();
}
//canny测试
void test_canny(){
cv::Mat img = cv::imread(SRC_PATH+"/img1.jpg",0);
cv::Mat opencv_canny;
cv::Canny(img,opencv_canny,100,150);
cv::namedWindow("opencv_canny",cv::WINDOW_NORMAL);
cv::resizeWindow("opencv_canny",img.size()/3);
cv::imshow("opencv_canny",opencv_canny);
cv::waitKey();

}

int main(){
    test_Sobel();
    // test_prewitt();
    // test_canny();
    return 0;
}