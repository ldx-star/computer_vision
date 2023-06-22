//
// Created by liangdaxin on 23-6-20.
//

#ifndef CANNY_EDGE_DETECTION_UTIL_H
#define CANNY_EDGE_DETECTION_UTIL_H
#include <opencv2/opencv.hpp>
#include <math.h>

class util {
public:
    static void Sobel(const cv::Mat& img, cv::Mat& x_derivative, cv::Mat& y_derivative);
    static void Prewitt(const cv::Mat& img, cv::Mat& x_derivative, cv::Mat& y_derivative);
    static void get_gradient(const cv::Mat& x_derivative,const cv::Mat& y_derivative, cv::Mat& magnitude, cv::Mat& direction );

};


#endif //CANNY_EDGE_DETECTION_UTIL_H
