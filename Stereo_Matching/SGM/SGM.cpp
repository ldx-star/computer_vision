//
// Created by liangdaxin on 23-6-19.
//

#include "SGM.h"
#include <cassert>

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
        census_left_ = cv::Mat::zeros(height, width, CV_32S);
        census_right_ = cv::Mat::zeros(height, width, CV_32S);
    }

    disp_left_ = cv::Mat::zeros(height, width, CV_32F);
    //视差范围
    const sint32 disp_range = option.max_disparity - option.min_disparity;
    if (disp_range <= 0) {
        return false;
    }
    uint32 size = width_ * height_ * disp_range;
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

    //视差计算
    SGM::ComputeDisparity();

//    cv::Mat img = disp_left_.clone();
//    img /= 255;
//    std::cout << img.size() << std::endl;
//    cv::imshow("img",img);
//    cv::waitKey();
}

void SGM::ComputeDisparity() {
    const sint32 min_disparity = option_.min_disparity;
    const sint32 max_disparity = option_.max_disparity;
    auto cost_ptr = cost_init_;
    const sint32 disp_range = max_disparity - min_disparity;

    //计算最优视差
    for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
            if(i == 316 && j == 387){
                int a = 10;
            }
            uint8 max_cost = 0;
            uint8 min_cost = INT8_MAX;
            uint8 best_disparity = 0;
            for (uint8 d = min_disparity; d < max_disparity; d++) {
                const auto &cost = cost_ptr[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
                if (min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
                max_cost = std::max(max_cost, cost);
            }
            //最小代价对应的视差值为最优视差
            if (max_cost != min_cost) {
                if (best_disparity != 0) {
                    int a = 10;
                }
                disp_left_.at<float>(i, j) = static_cast<float>(best_disparity);
            } else {
                //所有视差代价都一样则为无效视差
                disp_left_.at<float>(i, j) = 0;
            }
        }
    }
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
                if (j - d < 0 || j - d >= width_) {
                    cost = UINT8_MAX / 2;
                    continue;
                }
                if (option_.censusSize == Census5x5) {
                    const auto &census_val_l = census_left_.at<u_int32_t>(i, j);
                    const auto &census_val_r = census_right_.at<u_int32_t>(i, j-d);
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