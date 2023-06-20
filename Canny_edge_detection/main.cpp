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
    cv::namedWindow("x_derivative",cv::WINDOW_NORMAL);
    cv::namedWindow("y_derivative",cv::WINDOW_NORMAL);
    cv::resizeWindow("x_derivative",img.size()/3);
    cv::resizeWindow("y_derivative",img.size()/3);
    cv::imwrite(OUT_PATH+"/x_derivative.jpg",x_derivative);
    cv::imwrite(OUT_PATH+"/y_derivative.jpg",y_derivative);
    cv::imshow("x_derivative",x_derivative);
    cv::imshow("y_derivative",y_derivative);
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

int main(){
    test_Sobel();
    //test_prewitt();
    return 0;
}