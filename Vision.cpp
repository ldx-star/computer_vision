//
// Created by liangdaxin on 23-6-15.
//

#include "Vision.h"
#include <cmath>

cv::Mat Vision:: Gaussian_Kernel(const int &sigma, const int &width) {
    if (width % 2 == 0) {
        //暂时只接受奇数，后续完善
        std::cout << "width必须为奇数" << std::endl;
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

void Vision::Normalize(const cv::Mat &src, cv::Mat &des) {
    des = src;
    for(int i = 0; i < src.rows;i++){
        for(int j = 0; j < src.cols; j++){
            des.at<float>(i,j) = src.at<float>(i,j) / 255;
        }
    }
}

/*
 * canny边缘提取器
 * @param img
 * @param sigma
 * @param width
 */
cv::Mat Vision::Canny(const cv::Mat &img, const int &sigma, const int &width) {
    //获得gaussian_kernel
    cv::Mat gaussian_kernel = Vision::Gaussian_Kernel(sigma,width);
    //获得高斯偏导模板

    cv::Mat horizontal_gaussian_filter;
    cv::Mat vertical_gaussian_filter;
    cv::Sobel(gaussian_kernel, horizontal_gaussian_filter, CV_32F, 1, 0);
    cv::Sobel(gaussian_kernel, vertical_gaussian_filter, CV_32F, 0, 1);

    //用高斯偏导卷积模板对img卷积
    cv::Mat horizontal_img;
    cv::Mat vertical_img;
    cv::filter2D(img,horizontal_img,CV_32F,horizontal_gaussian_filter);
    cv::filter2D(img,vertical_img,CV_32F,vertical_gaussian_filter);
    cv::Mat square_horizontal_img;
    cv::Mat square_vertical_img;
    cv::pow(horizontal_img,2,square_horizontal_img);
    cv::pow(vertical_img,2,square_vertical_img);

    cv::Mat out_img;
    cv::sqrt(square_horizontal_img+square_vertical_img,out_img);

    //Non-maximum suppression
    cv::Mat gradient,angle;
    get_gradient_img(out_img,gradient,angle);

    //threshold
    //high threshold
    cv::Mat high_threshold_img = Threshold(out_img,0.1,1.0);
    //low threshold
    cv::Mat low_threshold_img = Threshold(out_img,0.05,1.0);


    cv::imshow("img",out_img);
    cv::imshow("high_threshold_img",high_threshold_img);
    cv::imshow("low_threshold_img",low_threshold_img);
    cv::waitKey();
    return low_threshold_img;

}

cv::Mat Vision::Threshold(const cv::Mat& img, double threshold, double max) {
    cv::Mat ret(img.rows,img.cols,CV_32F);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if(img.at<float>(i,j) < threshold){
                ret.at<float>(i,j) = 0;
            }else if(img.at<float>(i,j) > max){
                ret.at<float>(i,j) = max;
            }else{
                ret.at<float>(i,j) = img.at<float>(i,j);
            }
        }
    }
    return ret;
}

void Vision::get_gradient_img(const cv::Mat &img,cv::Mat& gradient, cv::Mat& angle) {
    cv::Mat x_kernel = (cv::Mat_<float>(1,2) << -1,1);
    cv::Mat y_kernel = (cv::Mat_<float>(2,1) << -1,1);
    cv::Mat new_img_x = cv::Mat::zeros(img.rows,img.cols,CV_32F);
    cv::Mat new_img_y = cv::Mat::zeros(img.rows,img.cols,CV_32F);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if(j == 0){
                new_img_y.at<float>(i,j) = 1.0;
            }else{
                new_img_y.at<float>(i,j) = img.at<float>(i,j-1)*y_kernel.at<float>(0,0) + img.at<float>(i,j)*y_kernel.at<float>(1,0);
            }
            if(i == 0){
                new_img_x.at<float>(i,j) = 1.0;
            }else{
                new_img_x.at<float>(i,j) = img.at<float>(i-1,j)*y_kernel.at<float>(0,0) + img.at<float>(i,j)*x_kernel.at<float>(0,1);
            }
        }
    }
    cv::cartToPolar(new_img_x,new_img_y,gradient,angle, true);

}