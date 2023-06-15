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

cv::Mat Vision::Covolution(const cv::Mat &img, const cv::Mat& kernel, const int& kernel_size) {
    if(kernel_size % 2 == 0){
        std::cout<<"kernel_size必须为奇数"<<std::endl;
        exit(1);
    }
    int num = kernel_size / 2;
    cv::Mat ret(img.rows-2*num,img.cols-2*num,CV_32F);
    for(int i = num; i < img.cols - num; i++){
        for(int j = num; j < img.rows - num; j++){

            float value = cal_value(img,kernel,kernel_size, j-num, i-num);
            ret.at<float>(j-num,i-num) = value;
        }
    }
    return ret;
}
/*
 * 计算卷积的值
 * @param img 图像
 * @param kernel 卷积核
 * @param kernel_size 卷积核大小
 * @param x,y 开始卷积的位子
 */
float Vision::cal_value(const cv::Mat& img, const cv::Mat& kernel, const int& kernel_size,int x,int y) {
    if(y==3){
        int a = 10;
    }
    float value = 0.0;
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            value += img.at<float>(x+i,y+j) * kernel.at<float>(i,j);
        }
    }
    return value;
}