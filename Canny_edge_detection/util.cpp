//
// Created by liangdaxin on 23-6-20.
//

#include "util.h"

void util::Sobel(const cv::Mat &img, cv::Mat &x_derivative, cv::Mat &y_derivative) {
    cv::Mat x_kernel = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat y_kernel = (cv::Mat_<int8_t>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    x_derivative = cv::Mat::zeros(img.size(), CV_8U);
    y_derivative = cv::Mat::zeros(img.size(), CV_8U);
    cv::filter2D(img, x_derivative, CV_8U, x_kernel);
    cv::filter2D(img, y_derivative, CV_8U, y_kernel);
}

void util::Prewitt(const cv::Mat &img, cv::Mat &x_derivative, cv::Mat &y_derivative) {
    cv::Mat x_kernel = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat y_kernel = (cv::Mat_<int8_t>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
    x_derivative = cv::Mat::zeros(img.size(), CV_8U);
    y_derivative = cv::Mat::zeros(img.size(), CV_8U);
    cv::filter2D(img, x_derivative, CV_8U, x_kernel);
    cv::filter2D(img, y_derivative, CV_8U, y_kernel);
}

void
util::get_gradient(const cv::Mat &x_derivative, const cv::Mat &y_derivative, cv::Mat &magnitude, cv::Mat &direction) {
    if(x_derivative.size() != y_derivative.size()){
        std::cout << "偏导矩阵size不一致"<<std::endl;
        return;
    }
    magnitude = cv::Mat::zeros(x_derivative.size(),CV_32F);
    direction = cv::Mat::zeros(x_derivative.size(),CV_32F);
    for(int i = 0;i < x_derivative.rows;i++){
        for(int j = 0; j < y_derivative.cols;j++){
            magnitude.at<float>(i,j) = sqrt(x_derivative.at<uint8_t>(i,j) * x_derivative.at<uint8_t>(i,j) + y_derivative.at<uint8_t>(i,j) * y_derivative.at<uint8_t>(i,j));
            if(x_derivative.at<uint8_t>(i,j) == 0){
                direction.at<float>(i,j) = M_PI / 2;
            }else{
                direction.at<float>(i,j) = atan(y_derivative.at<uint8_t>(i,j) / x_derivative.at<uint8_t>(i,j));
            }
        }

    }
}