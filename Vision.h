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
};


#endif //COMPUTER_VISION_VISION_H
