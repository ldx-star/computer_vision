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
    delete[] cost_aggr_2_;
    delete[] cost_aggr_3_;
    delete[] cost_aggr_4_;
    delete[] cost_aggr_5_;
    delete[] cost_aggr_6_;
    delete[] cost_aggr_7_;
    delete[] cost_aggr_8_;
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

    disp_left_ = new float32[height * width]();
    disp_right_ = new float32[height * width]();
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
    cost_aggr_5_ = new uint8[size]();
    cost_aggr_6_ = new uint8[size]();
    cost_aggr_7_ = new uint8[size]();
    cost_aggr_8_ = new uint8[size]();


    is_initialized_ = true;
}

void SGM::Show_disparity() {
    cv::Mat disp_img = cv::Mat::zeros(height_, width_, CV_8U);
    auto disp_data = disp_left_;
    float min_disp = width_;
    float max_disp = 0;
    for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
            float disp = disp_data[i * width_ + j];
            if (disp_data[i * width_ + j] != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
            float disp = disp_data[i * width_ + j];
            if (disp != Invalid_Float) {
                disp_img.at<uint8_t>(i, j) = int((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }
//    cv::imwrite("../out/not_use_MedianFilter.jpg",disp_img);
    cv::imshow("disp_img", disp_img);
    cv::waitKey();
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

    end = std::chrono::steady_clock::now();
    tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("cost aggregation! timimg:%lfs\n", tt.count() / 1000.0);

    start = std::chrono::steady_clock::now();

    //视差计算并保存
    SGM::ComputeDisparity(cost_aggr_);

    end = std::chrono::steady_clock::now();
    tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("disparity calculation! timimg:%lfs\n", tt.count() / 1000.0);

    start = std::chrono::steady_clock::now();


    //左右一致性检查
    if (option_.is_check_lr) {
        //计算右视差
        ComputeRightDisparity();
        //一致性检查
        LRCheck();
    }

    //去除最小连通区域
    if (option_.is_remove_speckles) {
        sgm_util::RemoveSpeckles(disp_left_, width_, height_, 1, option_.min_speckle_area);
    }

    if (option_.is_fill_holes) {
        FillHolesInDispMap();
    }

    //中值滤波
    sgm_util::MedianFilter(disp_left_, disp_left_, width_, height_, 3);

    end = std::chrono::steady_clock::now();
    tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("disparity optimization ! timimg:%lfs\n", tt.count() / 1000.0);



//    cv::Mat img = disp_left_.clone();
//    img /= 255;
//    std::cout << img.size() << std::endl;
//    cv::imshow("img",img);
//    cv::waitKey();
}


void SGM::FillHolesInDispMap() {
//    const int &width = width_;
//    const int &height = height_;
//    float32 *disparity_map = disp_left_;
//    auto &occlusions = occlusions_;//遮挡区像素集
//    auto &mismatches = mismatches_;//误匹配区像素集
//    std::vector<float> angles = {0, M_PI / 4, M_PI / 2, M_PI * 3 / 4, M_PI, M_PI * 5 / 4, M_PI * 3 / 2, M_PI * 7 / 4};
//
//    for (int k = 0; k < 3; k++) {
//        std::vector<float> angle_disp;//存放不同方向上碰到的第一个值
//
//        auto &disp_ptr = (k == 0) ? occlusions : mismatches;
//        //第一次循环处理遮挡区，第二次循环处理误匹配区，第三次循环处理剩下的区域
//        if (k == 2) {
//            disp_ptr.clear();
//            for (int i = 0; i < height; i++) {
//                for (int j = 0; j < width; j++) {
//                    if (disparity_map[i * width + j] == Invalid_Float) {
//                        disp_ptr.emplace_back(std::pair<int, int>(i, j));
//                    }
//                }
//            }
//        }
//        if (disp_ptr.empty()) {
//            continue;
//        }
//        for (auto pair: disp_ptr) {
//            int row = pair.first;
//            int col = pair.second;
//            angle_disp.clear();
//            //8个方向
//            for (auto angle: angles) {
//
//
//                auto sin = std::sin(angle);
//                auto cos = std::cos(angle);
//
//                for (int n = 1; n < 10; n++) {
//                    int y = 0, x = 0;
//                    //找到该方向上的第一个值
//                    y = lround(row + n * sin);
//                    x = lround(col + n * cos);
//                    //std::cout << x << " " << y << std::endl;
//                    if (y >= 0 && y < height && x >= 0 && x < width) {
//                        float disp = disparity_map[y * width + x];
//                        if (disp != Invalid_Float) {
//                            angle_disp.emplace_back(disp);
//                            break;
//                        }
//                    } else {
//                        break;
//                    }
//                }
//            }
//            if (angle_disp.empty()) {
//                continue;
//            }
//            std::sort(angle_disp.begin(), angle_disp.end());
//            //k==0时填充遮挡区
//            if (k == 0) {
//                //对于遮挡区像素，因为它的身份是背景像素，所以它是不能选择周围的前景像素视差值的，应该选择周围背景像素的视差值。由于背景像素视差值比前景像素小，
//                //所以在收集周围的有效视差值后，应选择较小的几个，具体哪一个呢？SGM作者选择的是次最小视差。
//                if (angle_disp.size() > 1) {
//                    disparity_map[row * width + col] = angle_disp[1];//次最小值
//                } else {
//                    disparity_map[row * width + col] = angle_disp[0];//只有一个值，就取最小值
//                }
//            } else {
//                //对于误匹配像素，它并不位于遮挡区，所以周围的像素都是可见的，而且没有遮挡导致的视差非连续的情况，
//                //它就像一个连续的表面凸起的一小块噪声，这时周围的视差值都是等价的，没有哪个应选哪个不应选，这时取中值就很适合
//                disparity_map[row * width + col] = angle_disp[angle_disp.size() / 2];//取中值
//            }
//        }
//
//    }

    const sint32 width = width_;
    const sint32 height = height_;

    std::vector<float32> disp_collects;

    // 定义8个方向
    const float32 pi = 3.1415926f;
    float32 angle1[8] = {pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4};
    float32 angle2[8] = {pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4};
    float32 *angle = angle1;
    // 最大搜索行程，没有必要搜索过远的像素
    const sint32 max_search_length = 1.0 * std::max(abs(option_.max_disparity), abs(option_.min_disparity));

    float32 *disp_ptr = disp_left_;
    for (sint32 k = 0; k < 3; k++) {
        // 第一次循环处理遮挡区，第二次循环处理误匹配区
        auto &trg_pixels = (k == 0) ? occlusions_ : mismatches_;
        if (trg_pixels.empty()) {
            continue;
        }
        std::vector<float32> fill_disps(trg_pixels.size());
        std::vector<std::pair<sint32, sint32>> inv_pixels;
        if (k == 2) {
            //  第三次循环处理前两次没有处理干净的像素
            for (sint32 i = 0; i < height; i++) {
                for (sint32 j = 0; j < width; j++) {
                    if (disp_ptr[i * width + j] == Invalid_Float) {
                        inv_pixels.emplace_back(i, j);
                    }
                }
            }
            trg_pixels = inv_pixels;
        }

        // 遍历待处理像素
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto &pix = trg_pixels[n];
            const sint32 y = pix.first;
            const sint32 x = pix.second;

            if (y == height / 2) {
                angle = angle2;
            }

            // 收集8个方向上遇到的首个有效视差值
            disp_collects.clear();
            for (sint32 s = 0; s < 8; s++) {
                const float32 ang = angle[s];
                const float32 sina = float32(sin(ang));
                const float32 cosa = float32(cos(ang));
                for (sint32 m = 1; m < max_search_length; m++) {
                    const sint32 yy = lround(y + m * sina);
                    const sint32 xx = lround(x + m * cosa);
                    if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
                        break;
                    }
                    const auto &disp = *(disp_ptr + yy * width + xx);
                    if (disp != Invalid_Float) {
                        disp_collects.push_back(disp);
                        break;
                    }
                }
            }
            if (disp_collects.empty()) {
                continue;
            }

            std::sort(disp_collects.begin(), disp_collects.end());

            // 如果是遮挡区，则选择第二小的视差值
            // 如果是误匹配区，则选择中值
            if (k == 0) {
                if (disp_collects.size() > 1) {
                    fill_disps[n] = disp_collects[1];
                } else {
                    fill_disps[n] = disp_collects[0];
                }
            } else {
                fill_disps[n] = disp_collects[disp_collects.size() / 2];
            }
        }
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto &pix = trg_pixels[n];
            const sint32 y = pix.first;
            const sint32 x = pix.second;
            disp_ptr[y * width + x] = fill_disps[n];
        }
    }

}

void SGM::LRCheck() {
    auto left_disparity = disp_left_;
    auto right_disparity = disp_right_;
    auto &occlusions = occlusions_;
    auto &mismatches = mismatches_;
    const uint32 width = width_;
    const uint32 height = height_;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int left_disp = static_cast<int>(left_disparity[i * width + j] + 0.5);// +0.5做到四舍五入
            int col_right = j - left_disp;
            if (col_right >= 0 && col_right < width) {
                int right_disp = static_cast<int>(right_disparity[i * width + col_right] + 0.5);
                if (abs(left_disp - right_disp) > option_.lrcheck_thres) {
                    //无效视差，判断是遮挡区还是误匹配区
                    int col_left = col_right + right_disp;
                    if (col_left >= 0 && col_left < width) {
                        int disp_l = static_cast<int>(left_disparity[i * width + col_right + right_disp] +
                                                      0.5);   //右视图在左图的匹配点
                        if (left_disp < disp_l) {
                            //遮挡区
                            occlusions.emplace_back(std::pair<int, int>(i, j));
                        } else {
                            mismatches.emplace_back(std::pair<int, int>(i, j));
                        }
                    }

                    left_disparity[i * width + j] = Invalid_Float;
                }
            } else {
                mismatches.emplace_back(std::pair<int, int>(i, j));
                left_disparity[i * width + j] = Invalid_Float;
            }
        }
    }
}

void SGM::ComputeRightDisparity() {
    auto min_disparity = option_.min_disparity;
    auto max_disparity = option_.max_disparity;
    sint32 disp_range = max_disparity - min_disparity;
    assert(disp_range > 0);
    uint32 width = width_;
    uint32 height = height_;

    const auto cost_aggr = cost_aggr_;
    const auto disparity = disp_right_;

    std::vector<uint8> cost_local(disp_range);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uint8 min_cost = UINT8_MAX;
            uint8 sec_min_cost = UINT8_MAX;
            sint32 best_disparity = 0;
            for (int d = min_disparity; d < max_disparity; d++) {
                const sint32 d_idx = d - min_disparity;
                const sint32 col_left = j + d;
                auto cost = cost_aggr + i * width * disp_range + col_left * disp_range;
                if (col_left > 0 && col_left < width) {
                    cost_local[d_idx] = cost[d_idx];
                    if (min_cost > cost[d_idx]) {
                        min_cost = cost[d_idx];
                        best_disparity = d;
                    }
                } else {
                    cost_local[d_idx] = UINT8_MAX;
                }
            }

            if (option_.is_check_unique) {
                //一个像素中只能有一个 最小代价和次小代价的差需要超过一个阈值
                for (int d = min_disparity; d < max_disparity; d++) {
                    if (d == best_disparity) {
                        //跳过最优视差
                        continue;
                    }
                    sec_min_cost = std::min(sec_min_cost, cost_local[d - min_disparity]);
                }
                if (sec_min_cost - min_cost <= static_cast<uint16>( float(min_cost) * (1 - option_.uniqueness_ratio))) {
                    disparity[i * width + j] = Invalid_Float;//无效视差
                    continue;
                }
            }
            //亚像素拟合
            if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
                disparity[i * width + j] = Invalid_Float;// 无效视差
                continue;
            } else {
                const sint32 idx_1 = best_disparity - 1 - min_disparity;
                const sint32 idx_2 = best_disparity + 1 - min_disparity;
                const sint32 cost1 = cost_local[idx_1];
                const sint32 cost2 = cost_local[idx_2];
                const uint16 denom = std::max(1, cost1 + cost2 - 2 * min_cost);
                disparity[i * width + j] = static_cast<float>(best_disparity) + (cost1 - cost2) / (2.0 * denom);
            }
        }
    }
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

    const auto &P1 = option_.p1;
    const auto &P2 = option_.p2;

    if (option_.num_paths == 4 || option_.num_paths == 8) {
        //左右聚合
        sgm_util::CostAggregateLeftRight(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                         cost_aggr_2_,
                                         false);
        sgm_util::CostAggregateLeftRight(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                         cost_aggr_1_,
                                         true);
        //上下聚合
        sgm_util::CostAggregateUpDown(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                      cost_aggr_4_,
                                      false);
        sgm_util::CostAggregateUpDown(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                      cost_aggr_3_,
                                      true);
    }

    if (option_.num_paths == 8) {
        //左上右下聚合
        sgm_util::CostAggregateDiagonal1(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                         cost_aggr_5_,
                                         true);
        sgm_util::CostAggregateDiagonal1(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                         cost_aggr_6_,
                                         false);
        //右上左下聚合
        sgm_util::CostAggregateDiagonal2(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                         cost_aggr_7_,
                                         true);
        sgm_util::CostAggregateDiagonal2(left_img_, width_, height_, min_disparity, max_disparity, P1, P2, cost_init_,
                                         cost_aggr_8_,
                                         false);
    }

    //把4/8个方向加起来
    for (sint32 i = 0; i < size; i++) {
        if (option_.num_paths == 4 || option_.num_paths == 8) {
            cost_aggr_[i] += cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
        }
        if (option_.num_paths == 8) {
            cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
        }
    }

}

void SGM::ComputeDisparity(uint8 *cost_ptr) {
    const sint32 min_disparity = option_.min_disparity;
    const sint32 max_disparity = option_.max_disparity;
    //auto cost_ptr = cost_aggr_2_;
    const sint32 disp_range = max_disparity - min_disparity;
    auto disparity = disp_left_;
    std::vector<uint8> cost_local(disp_range);
    uint32 height = height_;
    uint32 width = width_;
    //计算最优视差
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i == 1 && j == 6) {
                int a = 0;
            }
            uint8 max_cost = 0;
            uint8 min_cost = INT8_MAX;
            uint8 best_disparity = 0;
            for (uint8 d = min_disparity; d < max_disparity; d++) {
                const auto &cost = cost_local[d - min_disparity] = cost_ptr[i * width * disp_range + j * disp_range +
                                                                            (d - min_disparity)];
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
            if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
                disparity[i * width + j] = Invalid_Float;
                continue;
            } else {
                const sint32 idx_1 = best_disparity - 1 - min_disparity;
                const sint32 idx_2 = best_disparity + 1 - min_disparity;
                const sint32 cost1 = cost_local[idx_1];
                const sint32 cost2 = cost_local[idx_2];
                const uint16 denom = std::max(1, cost1 + cost2 - 2 * min_cost);
                float value = float(best_disparity) + (cost1 - cost2) / (2.0 * denom);
                disparity[i * width + j] = value;
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
                } else {
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