//
// Created by liangdaxin on 23-6-15.
//

#include "Vision.h"
#include <cmath>

cv::Mat Vision::Hough(const cv::Mat &img, int step) {
    cv::Mat hough_matrix = hough_algorithm(img,step);
    cv::Size size = hough_matrix.size();
    cv::namedWindow("hough_img",cv::WINDOW_NORMAL);
    cv::resizeWindow("hough_img",ceil(size.width/3),ceil(size.height/3));
    cv::imshow("hough_img",hough_matrix);
}

cv::Mat Vision:: hough_algorithm(const cv::Mat& img,int step){
    int _x = img.rows;
    int _y = img.cols;
    int _r = ceil(sqrt(_x*_x + _y*_y));
    cv::Mat hough_matrix = cv::Mat::zeros(_x,_y,_r);
    cv::Mat angle;
    cv::Mat gradient;
    get_gradient_img(img,gradient,angle);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if(img.at<int>(i,j) > 0) {
                int x = i;
                int y = j;
                int r = 0;
                while (angle.at<float>(i,j) < 1.5 && x < _x && y < _y && r < _r) {//当梯度方向为90度时不采纳
                    hough_matrix.at<int>(floor(x / step), floor(y / step), floor(r / step)) += 1;
                    x += step;
                    y += step * tan(angle.at<float>(i, j));
                    r += sqrt(pow(step * tan(angle.at<float>(i, j)), 2) + pow(step, 2));
                }
                x = i;
                y = j;
                r = 0;
                while(angle.at<float>(i,j) < 1.5 &&x >= 0 && y >= 0 && r >= 0){
                    hough_matrix.at<int>(x / step, y / step, r / step) += 1;
                    x -= step;
                    y -= floor(step * tan(angle.at<float>(i, j)));
                    r -= sqrt(pow(step * tan(angle.at<float>(i, j)), 2) + pow(step, 2));
                }
            }
        }
    }
    return hough_matrix;
}


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
    cv::Mat suppression_img = Non_maximum_suppression(out_img);
    //thresholding
    cv::Mat threshold_img = Thresholding1(suppression_img,0.05,0.1);

    return threshold_img;

}

cv::Mat Vision::Non_maximum_suppression(const cv::Mat &img) {
    cv::Mat gradient,angle;
    get_gradient_img(img,gradient,angle);

    cv::Mat ret = cv::Mat::zeros(img.rows,img.cols,CV_32F);
    for(int i = 1; i < img.rows; i++){
        for(int j = 1; j < img.cols;j++){
            float g1,g2,g3,g4;
            //偏向垂直方向
            if(abs(angle.at<float>(i,j)) < 1){
                g2 = img.at<float>(i-1,j);
                g4 = img.at<float>(i+1,j);
                if(angle.at<float>(i,j) > 0){
                    /*
                     *  g1 g2
                     *     c
                     *     g4 g3
                     */
                    g1 = img.at<float>(i-1,j-1);
                    g3 = img.at<float>(i+1,j+1);
                }else{
                    /*
                     *      g2 g1
                     *      c
                     *   g3 g4
                     */
                    g1 = img.at<float>(i-1,j+1);
                    g3 = img.at<float>(i+1,j-1);
                }
            }
            //偏水平方向
            else{
                g2 = img.at<float>(i,j-1);
                g4 = img.at<float>(i,j+1);
                if(angle.at<float>(i,j) > 0){
                    /*
                     *  g1
                     *  g2 c g4
                     *       g3
                     */
                    g1 = img.at<float>(i-1,j-1);
                    g3 = img.at<float>(i+1,j+1);
                }else{
                    /*
                     *       g3
                     *  g2 c g4
                     *  g1
                     */
                    g1 = img.at<float>(i+1,j-1);
                    g3 = img.at<float>(i-1,j+1);
                }
            }
            float temp1 = abs(angle.at<float>(i,j)) * g1 + (1-abs(angle.at<float>(i,j))) * g2;
            float temp2 = abs(angle.at<float>(i,j)) * g3 + (1-abs(angle.at<float>(i,j))) * g4;
            if(img.at<float>(i,j) >= temp1 && img.at<float>(i,j) >= temp2 ){
                ret.at<float>(i,j) = img.at<float>(i,j);
            }else{
                ret.at<float>(i,j) = 0;
            }
        }
    }
    return ret;
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
    cv::Mat x_kernel = (cv::Mat_<int>(1,2) << -1,1);
    cv::Mat y_kernel = (cv::Mat_<int>(2,1) << -1,1);
    cv::Mat new_img_x = cv::Mat::zeros(img.rows,img.cols,CV_32S);
    cv::Mat new_img_y = cv::Mat::zeros(img.rows,img.cols,CV_32S);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if(i == 491 && j == 165){
                int a = 10;
            }
            if(i == 0){
                new_img_y.at<int>(i,j) = 1;
            }else{
                new_img_y.at<int>(i,j) = img.at<uchar>(i-1,j) * y_kernel.at<int>(0,0) + img.at<uchar>(i,j)*y_kernel.at<int>(1,0);
            }
            if(j == 0){
                new_img_x.at<int>(i,j) = 1;
            }else{
                new_img_x.at<int>(i,j) = img.at<uchar>(i,j-1)*y_kernel.at<int>(0,0) + img.at<uchar>(i,j)*x_kernel.at<int>(0,1);
            }
        }
    }
    cartToPolar(new_img_x,new_img_y,gradient,angle);

}

void Vision::cartToPolar(cv::Mat &new_img_x, cv::Mat &new_img_y, cv::Mat &gradient, cv::Mat &angle) {
    int rows = new_img_x.rows;
    int cols = new_img_x.cols;
    gradient = cv::Mat::zeros(rows,cols,CV_32F);
    angle = cv::Mat::zeros(rows,cols,CV_32F);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(i == 491&& j == 164){
                int a = 10;
            }
           gradient.at<float>(i,j) = sqrt(new_img_x.at<int>(i,j)*new_img_x.at<int>(i,j) + new_img_y.at<int>(i,j)*new_img_y.at<int>(i,j));
           if(new_img_x.at<int>(i,j) != 0){
               angle.at<float>(i,j) = atan(new_img_y.at<int>(i,j)/new_img_x.at<int>(i,j));
           }else{
               angle.at<float>(i,j) = M_PI / 2;;
           }
        }
    }
}

//version1
cv::Mat Vision:: Thresholding1(const cv::Mat& origin_img,float low_threshold,float high_threshold){
    cv::Mat img = origin_img.clone();
    cv::Mat ret = cv::Mat::zeros(img.rows,img.cols,CV_32F);

    for(int i = 1 ; i < img.rows;i++){
        for(int j = 1;j < img.cols; j++){
            if(img.at<float>(i,j) >= high_threshold){//与高阈值相连的点
                ret.at<float>(i,j) = img.at<float>(i,j);
                if(img.at<float>(i-1,j-1) >= low_threshold && img.at<float>(i-1,j-1) < high_threshold){
                    ret.at<float>(i-1,j-1) = high_threshold;
                    img.at<float>(i-1,j-1) = high_threshold;
                }
                if(img.at<float>(i-1,j) >= low_threshold && img.at<float>(i-1,j) < high_threshold){
                    ret.at<float>(i-1,j) = high_threshold;
                    img.at<float>(i-1,j) = high_threshold;
                }
                if(img.at<float>(i-1,j+1) >= low_threshold && img.at<float>(i-1,j+1) < high_threshold){
                    ret.at<float>(i-1,j+1) = high_threshold;
                    img.at<float>(i-1,j+1) = high_threshold;
                }
                if(img.at<float>(i,j-1) >= low_threshold && img.at<float>(i,j-1) < high_threshold){
                    ret.at<float>(i,j-1) = high_threshold;
                    img.at<float>(i,j-1) = high_threshold;
                }
                if(img.at<float>(i,j+1) >= low_threshold && img.at<float>(i,j+1) < high_threshold){
                    ret.at<float>(i,j+1) = high_threshold;
                    img.at<float>(i,j+1) = high_threshold;
                }
                if(img.at<float>(i+1,j-1) >= low_threshold && img.at<float>(i+1,j-1) < high_threshold){
                    ret.at<float>(i+1,j-1) = high_threshold;
                    img.at<float>(i+1,j-1) = high_threshold;
                }
                if(img.at<float>(i+1,j) >= low_threshold && img.at<float>(i+1,j) < high_threshold){
                    ret.at<float>(i+1,j) = high_threshold;
                    img.at<float>(i+1,j) = high_threshold;
                }
                if(img.at<float>(i+1,j+1) >= low_threshold && img.at<float>(i+1,j+1) < high_threshold){
                    ret.at<float>(i+1,j+1) = high_threshold;
                    img.at<float>(i+1,j+1) = high_threshold;
                }
            }
        }
    }
    int count = count_pixel(ret);
    return ret;
}
//version2
cv::Mat Vision:: Thresholding(cv::Mat& img,float low_threshold,float high_threshold){
    cv::Mat gradient,angle;
    get_gradient_img(img,gradient,angle);

    cv::Mat ret = cv::Mat::zeros(img.rows,img.cols,CV_32F);
    cv::Mat ret1 = cv::Mat::zeros(img.rows,img.cols,CV_32F);

    for(int i = 1 ; i < img.rows;i++){
        for(int j = 1;j < img.cols; j++){
            if(img.at<float>(i,j) >= high_threshold){//与高阈值相连的点
                ret.at<float>(i,j) = img.at<float>(i,j);
                ret1.at<float>(i,j) = img.at<float>(i,j);
                if(abs(angle.at<float>(i,j)) < 1) {
                    //偏竖直方向
                    if (img.at<float>(i - 1, j) >= low_threshold && img.at<float>(i - 1, j) < high_threshold) {
                        ret.at<float>(i - 1, j) = high_threshold;
                        img.at<float>(i - 1, j) = high_threshold;
                    }
                    if (img.at<float>(i + 1, j) >= low_threshold && img.at<float>(i + 1, j) < high_threshold) {
                        ret.at<float>(i + 1, j) = high_threshold;
                        img.at<float>(i + 1, j) = high_threshold;
                    }
                    if (angle.at<float>(i, j) > 0) {
                        /*
                         *  g1 g2
                         *     c
                         *     g4 g3
                         */
                        if (img.at<float>(i - 1, j) >= low_threshold && img.at<float>(i - 1, j) < high_threshold) {
                            ret.at<float>(i - 1, j - 1) = high_threshold;
                            img.at<float>(i - 1, j - 1) = high_threshold;
                        }
                        if (img.at<float>(i + 1, j + 1) >= low_threshold && img.at<float>(i + 1, j + 1) < high_threshold) {
                            ret.at<float>(i + 1, j + 1) = high_threshold;
                            img.at<float>(i + 1, j + 1) = high_threshold;
                        }
                    } else {
                        /*
                         *     g2 g1
                         *     c
                         *  g3 g4
                         */
                        if (img.at<float>(i - 1, j + 1) >= low_threshold && img.at<float>(i - 1, j + 1) < high_threshold) {
                            ret.at<float>(i - 1, j + 1) = high_threshold;
                            img.at<float>(i - 1, j + 1) = high_threshold;
                        }
                        if (img.at<float>(i + 1, j - 1) >= low_threshold && img.at<float>(i + 1, j - 1) < high_threshold) {
                            ret.at<float>(i + 1, j - 1) = high_threshold;
                            img.at<float>(i + 1, j - 1) = high_threshold;
                        }
                    }
                }else{
                    //偏水平方向
                    if(img.at<float>(i,j-1) >= low_threshold && img.at<float>(i,j-1) < high_threshold){
                        ret.at<float>(i, j - 1) = high_threshold;
                        img.at<float>(i, j - 1) = high_threshold;
                    }
                    if(img.at<float>(i,j+1) >= low_threshold && img.at<float>(i,j+1) < high_threshold){
                        ret.at<float>(i, j + 1) = high_threshold;
                        img.at<float>(i, j + 1) = high_threshold;
                    }
                    if(angle.at<float>(i,j) > 0){
                        /*
                         *   g1
                         *   g2 c g4
                         *        g3
                         */
                        if(img.at<float>(i-1,j-1) >= low_threshold && img.at<float>(i-1,j-1) < high_threshold){
                            ret.at<float>(i-1,j-1) = high_threshold;
                            img.at<float>(i-1,j-1) = high_threshold;
                        }
                        if(img.at<float>(i+1,j+1) >= low_threshold && img.at<float>(i+1,j+1) < high_threshold){
                            ret.at<float>(i+1,j+1) = high_threshold;
                            img.at<float>(i+1,j+1) = high_threshold;
                        }
                    }else{
                        /*
                         *        g3
                         *   g2 c g4
                         *   g1
                         */
                        if(img.at<float>(i+1,j-1) >= low_threshold && img.at<float>(i+1,j-1) < high_threshold){
                            ret.at<float>(i+1,j-1) = high_threshold;
                            img.at<float>(i+1,j-1) = high_threshold;
                        }
                        if(img.at<float>(i-1,j+1) >= low_threshold && img.at<float>(i-1,j+1) < high_threshold){
                            ret.at<float>(i-1,j+1) = high_threshold;
                            img.at<float>(i-1,j+1) = high_threshold;
                        }
                    }
                }
            }
        }
    }
    return ret;
}
int Vision::count_pixel(const cv::Mat& img){
    int count = 0;
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if(img.at<float>(i,j) > 0){
                count++;
            }
        }
    }
    return count;
}
