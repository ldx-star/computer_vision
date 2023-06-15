//
// Created by liangdaxin on 23-6-15.
//

#include "Vision.h"
#include <cmath>

cv::Mat Vision:: Gaussian_Kernel(const int &sigma, const int &width) {
    if (width % 2 == 0) {
        //暂时只接受奇数，后续完善
        std::cout << "width不能为奇数" << std::endl;
        exit(1);
    }
    int num = width / 2;
    cv::Mat G(width,width,CV_32F);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float value = (1 / (2 * M_PI * std::pow(sigma, 2))) * (std::pow(M_E, -1 * (std::pow(i - num, 2) +
                                                                                             std::pow(j - num, 2) )/ (2 *
                                                                                             std::pow(sigma, 2))));
            G.at<float>(i, j) = value;

        }
    }
    return G;
}