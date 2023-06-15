//
// Created by liangdaxin on 23-6-15.
//
#include "Vision.h"
void test(){
    cv::Mat img = cv::imread("./img1.jpeg",cv::IMREAD_GRAYSCALE);
    img.convertTo(img,CV_32F);
    cv::Mat kernel(3,3,CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(j == 0){
                kernel.at<float>(i,j) = -1.0;
            }else if(j == 1){
                kernel.at<float>(i,j) = 0;
            }else{
                kernel.at<float>(i,j) = 1.0;
            }
        }
    }
    cv::Mat out = Vision::Covolution(img,kernel,3);
}
void test1(){
    cv::Mat matrix(5, 7, CV_32F, cv::Scalar(0));

    // 设置第4列的元素为1
    for (int row = 0; row < matrix.rows; ++row)
    {
        matrix.at<float>(row, 3) = 1;
    }

    cv::Mat horizontalKernel = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat result;
    cv::filter2D(matrix, result, CV_32F, horizontalKernel);

    std::cout << "Result:\n" << result << std::endl;
}
int main(){
    test();

    return 0;
}