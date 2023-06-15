//
// Created by liangdaxin on 23-6-15.
//
#include "Vision.h"
//prewitt
cv::Mat test(){
    cv::Mat img = cv::imread("./img1.jpeg",cv::IMREAD_GRAYSCALE);
    img.convertTo(img,CV_32F);
    Vision::Normalize(img,img);
    cv::Mat horizontal_kernel(3,3,CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(j == 0){
                horizontal_kernel.at<float>(i,j) = -1.0;
            }else if(j == 1){
                horizontal_kernel.at<float>(i,j) = 0;
            }else{
                horizontal_kernel.at<float>(i,j) = 1.0;
            }
        }
    }
    cv::Mat vertical_kernel(3,3,CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(i == 0){
                vertical_kernel.at<float>(i,j) = 1.0;
            }else if(i == 1){
                vertical_kernel.at<float>(i,j) = 0;
            }else{
                vertical_kernel.at<float>(i,j) = -1.0;
            }
        }
    }
    cv::Mat out1 = Vision::Covolution(img,vertical_kernel,3);
    cv::Mat out2 = Vision::Covolution(img,horizontal_kernel,3);
    cv::Mat square_out1;
    cv::Mat square_out2;
    cv::pow(out1,2,square_out1);
    cv::pow(out2,2,square_out2);
    cv::Mat out;
    cv::sqrt(square_out1+square_out2,out);
    return out;
//    cv::imshow("img1",out1);
//    cv::imshow("img2",out2);
//    cv::imshow("img3",out);
//    cv::waitKey();
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
//高斯卷积
void test3(){
    cv::Mat gaussion_kernel = Vision::Gaussian_Kernel(2,5);
    cv::Mat img = cv::imread("./img1.jpeg",cv::IMREAD_GRAYSCALE);
    img.convertTo(img,CV_32F);
    cv::Mat nor_img;
    Vision::Normalize(img,nor_img);
    cv::Mat out = Vision::Covolution(img,gaussion_kernel,5);
    cv::imshow("origin",img);
    cv::imshow("out",out);
    cv::waitKey();
}
//canny
cv::Mat test4(){
    //高斯偏导模板
    cv::Mat gaussion_kernel = Vision::Gaussian_Kernel(1,5);
    cv::Mat horizontal_kernel(3,3,CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(j == 0){
                horizontal_kernel.at<float>(i,j) = -1.0;
            }else if(j == 1){
                horizontal_kernel.at<float>(i,j) = 0;
            }else{
                horizontal_kernel.at<float>(i,j) = 1.0;
            }
        }
    }
    cv::Mat vertical_kernel(3,3,CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(i == 0){
                vertical_kernel.at<float>(i,j) = 1.0;
            }else if(i == 1){
                vertical_kernel.at<float>(i,j) = 0;
            }else{
                vertical_kernel.at<float>(i,j) = -1.0;
            }
        }
    }
    cv::Mat out1 = Vision::Covolution(gaussion_kernel,vertical_kernel,3);
    cv::Mat out2 = Vision::Covolution(gaussion_kernel,horizontal_kernel,3);
//    cv::Mat square_out1;
//    cv::Mat square_out2;
//    cv::pow(out1,2,square_out1);
//    cv::pow(out2,2,square_out2);
//    cv::Mat gaussian_pd; // 高斯偏导模板
//    cv::sqrt(square_out1+square_out2,gaussian_pd);

    cv::Mat img = cv::imread("./img1.jpeg",cv::IMREAD_GRAYSCALE);
    img.convertTo(img,CV_32F);
    Vision::Normalize(img,img);
    cv::Mat img_out1 = Vision::Covolution(img,out1,out1.rows);
    cv::Mat img_out2 = Vision::Covolution(img,out2,out2.rows);
    cv::Mat square_out1,square_out2;
    cv::pow(img_out1,2,square_out1);
    cv::pow(img_out2,2,square_out2);
    cv::Mat img_out;
    cv::sqrt(square_out1+square_out2,img_out);
    return img_out;
//    cv::imshow("img1",img_out1);
//    cv::imshow("img2",img_out2);
//    cv::imshow("img",img_out);
//    cv::waitKey();
}
//canny 先高斯在求导
cv::Mat test5(){
    cv::Mat img = cv::imread("./img1.jpeg",cv::IMREAD_GRAYSCALE);
    img.convertTo(img,CV_32F);
    Vision::Normalize(img,img);
    cv::Mat gaussian_kernel = Vision::Gaussian_Kernel(1,7);
    cv::Mat img_gaussian = Vision::Covolution(img,gaussian_kernel,gaussian_kernel.rows);

    cv::Mat horizontal_kernel(3,3,CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(j == 0){
                horizontal_kernel.at<float>(i,j) = -1.0;
            }else if(j == 1){
                horizontal_kernel.at<float>(i,j) = 0;
            }else{
                horizontal_kernel.at<float>(i,j) = 1.0;
            }
        }
    }
    cv::Mat vertical_kernel(3,3,CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(i == 0){
                vertical_kernel.at<float>(i,j) = 1.0;
            }else if(i == 1){
                vertical_kernel.at<float>(i,j) = 0;
            }else{
                vertical_kernel.at<float>(i,j) = -1.0;
            }
        }
    }
    cv::Mat out1 = Vision::Covolution(img_gaussian,vertical_kernel,3);
    cv::Mat out2 = Vision::Covolution(img_gaussian,horizontal_kernel,3);
    cv::Mat square_out1;
    cv::Mat square_out2;
    cv::pow(out1,2,square_out1);
    cv::pow(out2,2,square_out2);
    cv::Mat out;
    cv::sqrt(square_out1+square_out2,out);
    return out;
//    cv::imshow("img1",out1);
//    cv::imshow("img2",out2);
//    cv::imshow("img3",out);
//    cv::waitKey();


}
int main(){
    cv::Mat img1 = test();
    cv::Mat img2 = test5();
    cv::Mat img3 = test4();
    cv::imshow("img1",img1);
    cv::imshow("img2",img2);
    cv::imshow("img3",img3);
    cv::waitKey();
    return 0;
}