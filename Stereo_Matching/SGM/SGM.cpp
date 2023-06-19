//
// Created by liangdaxin on 23-6-19.
//

#include "SGM.h"
SGM::SGM() {

}
SGM::~SGM() {}

bool SGM::Initialize(const uint32 &width, const uint32 &height, const SGM::SGMOption &option) {
    width_ = width;
    height_ = height;
    option_ = option;
    if(width_ == 0 || height_ == 0){
        return false;
    }

    //census值
    if(option_.censusSize == Census5x5){
        census_left_ = cv::Mat::zeros(left_img_.size(),CV_32S);
        census_right_ = cv::Mat::zeros(right_img_.size(),CV_32S);
    }

    //视差范围
    const sint32 disp_range = option.max_disparity - option.min_disparity;
    if(disp_range <= 0){
        return false;
    }

    is_initialized_ = true;
}
bool SGM::Match(const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& disp_left) {
    if(!is_initialized_){
        std::cout<<"未初始化"<<std::endl;
        return false;
    }
    left_img_ = img_left;
    right_img_ = img_right;

    auto start = std::chrono::steady_clock::now();

    //census变换
    CensusTransform();

}

void SGM::CensusTransform(){
    if(option_.censusSize == Census5x5){
        sgm_util::census_transform_5x5(left_img_,census_left_, width_,height_);
        sgm_util::census_transform_5x5(right_img_,census_right_, width_,height_);
    }

}