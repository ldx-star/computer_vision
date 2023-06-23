//
// Created by liangdaxin on 23-6-19.
//

#include "SGM.h"
#include <cassert>
#include <chrono>
#include <algorithm>
#include <vector>

SGM::SGM() {

}

SGM::~SGM() {
    delete[] cost_init_;
    delete[] cost_aggr_;
    delete[] cost_aggr_1_;
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
    cost_aggr_ = new uint8[size]();
    cost_aggr_1_ = new uint8[size]();
    cost_aggr_2_ = new uint8[size]();
    cost_aggr_3_ = new uint8[size]();
    cost_aggr_4_ = new uint8[size]();

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

    auto end = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("computing cost! timimg:%lfs\n", tt.count() / 1000.0);
    start = std::chrono::steady_clock::now();

    //聚合代价
    CostAggregation();

    //视差计算
    SGM::ComputeDisparity(cost_aggr_);
    sgm_util::save_img(disp_left_,"../out/cost_aggr.jpg");
    SGM::ComputeDisparity(cost_aggr_1_);
    sgm_util::save_img(disp_left_,"../out/aggr1.jpg");
    SGM::ComputeDisparity(cost_aggr_2_);
    sgm_util::save_img(disp_left_,"../out/aggr2.jpg");
    SGM::ComputeDisparity(cost_aggr_3_);
    sgm_util::save_img(disp_left_,"../out/aggr3.jpg");
    SGM::ComputeDisparity(cost_aggr_4_);
    sgm_util::save_img(disp_left_,"../out/aggr4.jpg");


//    cv::Mat img = disp_left_.clone();
//    img /= 255;
//    std::cout << img.size() << std::endl;
//    cv::imshow("img",img);
//    cv::waitKey();
}

void SGM::CostAggregation() {
    //聚合路径
    //1、左->右/右->左
    const auto &min_disparity = option_.min_disparity;
    const auto &max_disparity = option_.max_disparity;
    const sint32 disp_range = max_disparity - min_disparity;
    assert(disp_range > 0);
    const sint32 size = width_ * height_ * disp_range;
    assert(size >= 0);

    const auto& P1 = option_.p1;
    const auto& P2 = option_.p2;

    if(option_.num_paths == 4 || option_.num_paths == 8){
        //左右聚合
        sgm_util::CostAggregateLeftRight(left_img_,width_,height_,min_disparity,max_disparity,P1,P2,cost_init_,cost_aggr_2_,
                                         false);
        sgm_util::CostAggregateLeftRight(left_img_,width_,height_,min_disparity,max_disparity,P1,P2,cost_init_,cost_aggr_1_,
                                         true);
        //上下聚合
        sgm_util::CostAggregateUpDown(left_img_,width_,height_,min_disparity,max_disparity,P1,P2,cost_init_,cost_aggr_4_,
                                         false);
        sgm_util::CostAggregateUpDown(left_img_,width_,height_,min_disparity,max_disparity,P1,P2,cost_init_,cost_aggr_3_,
                                      true);
    }

    if(option_.num_paths == 8){
        //左上右下聚合

        //右上左下聚合
    }

    //把4/8个方向加起来
    for(sint32 i = 0; i < size;i++){
        cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
    }
}

void SGM::ComputeDisparity(uint8* cost_ptr) {
    const sint32 min_disparity = option_.min_disparity;
    const sint32 max_disparity = option_.max_disparity;
    //auto cost_ptr = cost_aggr_2_;
    const sint32 disp_range = max_disparity - min_disparity;

    //计算最优视差
    for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
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
//            //最小代价对应的视差值为最优视差
//            if (max_cost != min_cost) {
//                disp_left_.at<uint8_t>(i, j) = best_disparity;
//            } else {
//                //所有视差代价都一样则为无效视差
//                disp_left_.at<uint8_t>(i, j) = 0;
//            }

            //亚像素拟合
            if(best_disparity == min_disparity || best_disparity == max_disparity- 1){
                disp_left_.at<float>(i,j) = 0;
                continue;
            }else{
                const sint32 idx_1 = best_disparity - 1 -min_disparity;
                const sint32 idx_2 = best_disparity + 1 -min_disparity;
                const sint32 cost1 = cost_ptr[idx_1];
                const sint32 cost2 = cost_ptr[idx_2];
                const uint16 denom = std::max(1,cost1+cost2-2*min_cost );
                disp_left_.at<float>(i,j) = float(best_disparity) + (cost1-cost2)/(2.0 * denom);
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
    int count = 1;
    for (int i = 0; i < height_; i++) {

        for (int j = 0; j < width_; j++) {
            for (int d = min_disparity; d < max_disparity; d++) {
                auto &cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
                if (j - d < 0 || j - d >= width_) {
                    cost = UINT8_MAX / 2;
                    continue;
                }else{
                    const auto &census_val_l = census_left_.at<u_int32_t>(i, j);
                    const auto &census_val_r = census_right_.at<u_int32_t>(i, j - d);
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