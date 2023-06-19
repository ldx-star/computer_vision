//
// Created by liangdaxin on 23-6-19.
//

#include "SGM.h"

SGM::SGM() {

}

SGM::~SGM() {
    delete[] cost_init_;
}

bool SGM::Initialize(const uint32 &width, const uint32 &height, const SGM::SGMOption &option) {
    width_ = width;
    height_ = height;
    option_ = option;
    if (width_ == 0 || height_ == 0) {
        return false;
    }

    //census值
    if (option_.censusSize == Census5x5) {
        census_left_ = cv::Mat::zeros(width_,height_, CV_32S);
        census_right_ = cv::Mat::zeros(width_,height_, CV_32S);
    }

    //视差范围
    const sint32 disp_range = option.max_disparity - option.min_disparity;
    if (disp_range <= 0) {
        return false;
    }
    uint32 size = width_* height_ * disp_range;
    cost_init_ = new uint8[size]();

    is_initialized_ = true;
}

bool SGM::Match(const cv::Mat &img_left, const cv::Mat &img_right, cv::Mat &disp_left) {
    if (!is_initialized_) {
        std::cout << "未初始化" << std::endl;
        return false;
    }
    left_img_ = img_left;
    right_img_ = img_right;

    auto start = std::chrono::steady_clock::now();

    //census变换
    CensusTransform();

    //代价计算
    ComputeCost();

}

void SGM::ComputeCost() {
    const sint32 &min_disparity = option_.min_disparity;
    const sint32 &max_disparity = option_.max_disparity;
    const sint32 disp_range = max_disparity - min_disparity;
    if (disp_range <= 0) {
        return;
    }
    //计算代价（基于Hamming距离）
    for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
            for (int d = min_disparity; d < max_disparity; d++) {
                auto &cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
                if(i == 4 && j==379 && d == 46){
                    int a =10;
                }
                if (j - d < 0 || j - d >= width_) {
                    cost = UINT8_MAX / 2;
                    continue;
                }
                if (option_.censusSize == Census5x5) {
                    const auto &census_val_l = census_left_.at<u_int8_t>(i, j);
                    const auto &census_val_r = census_right_.at<u_int8_t>(i, j);

                    cost = sgm_util::Hamming32(census_val_l, census_val_r);
                }

            }
        }
    }

}

void SGM::CensusTransform() {
    if (option_.censusSize == Census5x5) {
        sgm_util::census_transform_5x5(left_img_, census_left_, width_, height_);
        sgm_util::census_transform_5x5(right_img_, census_right_, width_, height_);
    }

}