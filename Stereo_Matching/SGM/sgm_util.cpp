//
// Created by liangdaxin on 23-6-19.
//

#include "sgm_util.h"
#include <algorithm>

void sgm_util::census_transform_5x5(const cv::Mat &img, cv::Mat &census, const sint32 &width, const sint32 &height) {
    census = cv::Mat::zeros(img.size(), CV_32S);
    for (int i = 2; i < height - 2; i++) {
        for (int j = 2; j < width - 2; j++) {
            const uint8 center = img.at<u_int8_t>(i, j);
            sint32 census_value = 0;
            //遍历center的邻域
            for (int r = -2; r < 2; r++) {
                for (int c = -2; c < 2; c++) {
                    census_value <<= 1;
                    if (img.at<u_int8_t>(i + r, j + c) < center) {
                        census_value += 1;
                    }
                }
            }
            census.at<u_int32_t>(i, j) = census_value;
        }
    }
}

void sgm_util::save_img(cv::Mat &img, const std::string &filename) {
    img.convertTo(img, CV_8U);
    cv::imwrite(filename, img);
}

uint32 sgm_util::Hamming32(const uint32 &x, const uint32 &y) {
    uint32 distance = 0;
    uint32 val = x ^ y;
    //统计val中1个数
    while (val) {
        if (val & 1) {
            distance++;
        }
        val >>= 1;
    }
    return distance;
}

void sgm_util::CostAggregateLeftRight(const cv::Mat &img_data, const sint32 &width, const sint32 &height,
                                      const sint32 &min_disparity, const sint32 &max_disparity, const sint32 &p1,
                                      const sint32 &p2_init, const uint8 *cost_init, uint8 *cost_aggr,
                                      bool is_forward) {
    //视差范围
    const sint32 disp_range = max_disparity - min_disparity;
    //P1，P2
    const auto &P1 = p1;
    const auto &P2_init = p2_init;
    // 正向(左->右) ：is_forward = true ; direction = 1
    // 反向(右->左) ：is_forward = false; direction = -1
    const sint32 direction = is_forward ? 1 : -1;

    //聚合
    for (sint32 i = 0; i < height; i++) {
        //每一行的首列元素
        auto cost_init_row = (is_forward) ? (cost_init + i * width * disp_range) : (cost_init + i * width * disp_range +
                                                                                    (width - 1) * disp_range);
        auto cost_aggr_row = (is_forward) ? (cost_aggr + i * width * disp_range) : (cost_aggr + i * width * disp_range +
                                                                                    (width - 1) * disp_range);

        // 路径当前的灰度值和上一个灰度值
        uint8 gray = (is_forward) ? img_data.at<uint8_t>(i, 0) : img_data.at<uint8_t>(i, width - 1);
        uint8 last_gray = (is_forward) ? img_data.at<uint8_t>(i, 0) : img_data.at<uint8_t>(i, width - 1);

        //上一个元素的代价数组
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        //初始化：第一个像素的聚合代价等于初始代价
        memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));
        cost_init_row += direction * disp_range;
        cost_aggr_row += direction * disp_range;

        //上一个路径的最小代价
        uint8 minCost_lastPath = UINT8_MAX;
        for (auto cost: cost_last_path) {
            minCost_lastPath = std::min(minCost_lastPath, cost);
        }

        //从第二个元素开始聚合
        for (int j = 1; j < width; j++) {
            gray = (is_forward) ? img_data.at<uint8_t>(i, j) : img_data.at<uint8_t>(i, width - 1 - j);
            uint8 min_cost = UINT8_MAX;
            for (int d = 0; d < disp_range; d++) {
                // l1 = L(p-r,d)
                // l2 = L(p-r,d-1) + p1
                // l3 = L(p-r,d+1) + p1
                // l4 = min L(p-r) + p2
                const uint8 cost = cost_init_row[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + P1;
                const uint16 l3 = cost_last_path[d + 2] + P1;
                const uint16 l4 = minCost_lastPath + std::max(P1, P2_init / (abs(gray - last_gray) + 1));

                const uint8 cost_s =
                        cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - minCost_lastPath);

                cost_aggr_row[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }
            // 重置上一个元素的最小代价数组
            minCost_lastPath = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));

            // 下一个元素
            cost_init_row += direction * disp_range;
            cost_aggr_row += direction * disp_range;

            last_gray = gray;

        }
    }
}

void sgm_util::CostAggregateUpDown(const cv::Mat &img_data, const sint32 &width, const sint32 &height,
                                   const sint32 &min_disparity, const sint32 &max_disparity, const sint32 &p1,
                                   const sint32 &p2_init, const uint8 *cost_init, uint8 *cost_aggr, bool is_forward) {
    //视差范围
    const sint32 disp_range = max_disparity - min_disparity;
    //P1，P2
    const auto &P1 = p1;
    const auto &P2_init = p2_init;
    // 正向(上->下) ：is_forward = true ; direction = 1
    // 反向(下->上) ：is_forward = false; direction = -1
    const sint32 direction = is_forward ? 1 : -1;

    //聚合
    for (sint32 i = 0; i < width; i++) {
        //每一行的首列元素
        auto cost_init_row = (is_forward) ? (cost_init + i * disp_range) : (cost_init +
                                                                            width * disp_range * (height - 1) +
                                                                            i * disp_range);
        auto cost_aggr_row = (is_forward) ? (cost_aggr + i * disp_range) : (cost_aggr +
                                                                            width * disp_range * (height - 1) +
                                                                            i * disp_range);

        // 路径当前的灰度值和上一个灰度值
        uint8 gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);
        uint8 last_gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);

        //上一个元素的代价数组
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        //初始化：第一个像素的聚合代价等于初始代价
        memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));
        cost_init_row += direction * width * disp_range;
        cost_aggr_row += direction * width * disp_range;

        //上一个路径的最小代价
        uint8 minCost_lastPath = UINT8_MAX;
        for (auto cost: cost_last_path) {
            minCost_lastPath = std::min(minCost_lastPath, cost);
        }

        //从第二个元素开始聚合
        for (int j = 1; j < height; j++) {
            gray = (is_forward) ? img_data.at<uint8_t>(j, i) : img_data.at<uint8_t>(height - 1 - j, i);
            uint8 min_cost = UINT8_MAX;
            for (int d = 0; d < disp_range; d++) {
                // l1 = L(p-r,d)
                // l2 = L(p-r,d-1) + p1
                // l3 = L(p-r,d+1) + p1
                // l4 = min L(p-r) + p2
                const uint8 cost = cost_init_row[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + P1;
                const uint16 l3 = cost_last_path[d + 2] + P1;
                const uint16 l4 = minCost_lastPath + std::max(P1, P2_init / (abs(gray - last_gray) + 1));

                const uint8 cost_s =
                        cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - minCost_lastPath);

                cost_aggr_row[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }
            // 重置上一个元素的最小代价数组
            minCost_lastPath = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));

            // 下一个元素
            cost_init_row += direction * width * disp_range;
            cost_aggr_row += direction * width * disp_range;

            last_gray = gray;

        }
    }
}