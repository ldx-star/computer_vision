//
// Created by liangdaxin on 23-6-15.
//

#ifndef COMPUTER_VISION_VISION_H
#define COMPUTER_VISION_VISION_H
#include <iostream>
#include <opencv2/opencv.hpp>

class Vision {
public:
    static cv::Mat Gaussian_Kernel(const int& sigma, const int& width);
    static cv::Mat Covolution(const cv::Mat& img, const cv::Mat& kernel, const int& kernel_size);
    static void Normalize(const cv::Mat& src, cv::Mat& des);
    static cv::Mat Canny(const cv::Mat& img,const int& sigma, const int& width);
private:
    static float cal_value(const cv::Mat& img, const cv::Mat& kernel, const int& kernel_size, int x,int y);
    static cv::Mat Threshold(const cv::Mat& img,double threshold,double max);
    static void get_gradient_img(const cv::Mat& img, cv::Mat& gradient, cv::Mat& angle);
};


#endif //COMPUTER_VISION_VISION_H
