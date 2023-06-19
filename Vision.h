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
    static cv::Mat Non_maximum_suppression(const cv::Mat& img);
    static cv::Mat Thresholding(cv::Mat& img,float low_threshold,float high_threshold);
    static cv::Mat Thresholding1(const cv::Mat& img,float low_threshold,float high_threshold);
    static cv::Mat Hough(const cv::Mat& img,int step);
private:
    static void cartToPolar(cv::Mat& new_img_x,cv::Mat& new_img_y, cv::Mat& gradient, cv::Mat& angle);
    static float cal_value(const cv::Mat& img, const cv::Mat& kernel, const int& kernel_size, int x,int y);
    static cv::Mat Threshold(const cv::Mat& img,double threshold,double max);
    static void get_gradient_img(const cv::Mat& img, cv::Mat& gradient, cv::Mat& angle);
    static int count_pixel(const cv::Mat& img);
    static cv::Mat hough_algorithm(const cv::Mat& img,int step);

};


#endif //COMPUTER_VISION_VISION_H
