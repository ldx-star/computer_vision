//
// Created by liangdaxin on 23-6-20.
//

#include "util.h"

void util::Sobel(const cv::Mat &img, cv::Mat& x_derivative, cv::Mat& y_derivative) {
    cv::Mat x_kernel = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat y_kernel = (cv::Mat_<int8_t>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    x_derivative = cv::Mat::zeros(img.size(),CV_8U);
    y_derivative = cv::Mat::zeros(img.size(),CV_8U);
    cv::filter2D(img, x_derivative, CV_8U, x_kernel);
    cv::filter2D(img, y_derivative, CV_8U, y_kernel);
}
void util::Prewitt(const cv::Mat &img, cv::Mat &x_derivative, cv::Mat &y_derivative) {
    cv::Mat x_kernel = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat y_kernel = (cv::Mat_<int8_t>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
    x_derivative = cv::Mat::zeros(img.size(),CV_8U);
    y_derivative = cv::Mat::zeros(img.size(),CV_8U);
    cv::filter2D(img, x_derivative, CV_8U, x_kernel);
    cv::filter2D(img, y_derivative, CV_8U, y_kernel);
}