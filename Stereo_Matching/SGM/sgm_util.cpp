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
    //img.convertTo(img, CV_8U);
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
        //每一列的首列元素
        auto cost_init_col = (is_forward) ? (cost_init + i * disp_range) : (cost_init +
                                                                            width * disp_range * (height - 1) +
                                                                            i * disp_range);
        auto cost_aggr_col = (is_forward) ? (cost_aggr + i * disp_range) : (cost_aggr +
                                                                            width * disp_range * (height - 1) +
                                                                            i * disp_range);

        // 路径当前的灰度值和上一个灰度值
        uint8 gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);
        uint8 last_gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);

        //上一个元素的代价数组
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        //初始化：第一个像素的聚合代价等于初始代价
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));
        cost_init_col += direction * width * disp_range;
        cost_aggr_col += direction * width * disp_range;

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
                const uint8 cost = cost_init_col[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + P1;
                const uint16 l3 = cost_last_path[d + 2] + P1;
                const uint16 l4 = minCost_lastPath + std::max(P1, P2_init / (abs(gray - last_gray) + 1));

                const uint8 cost_s =
                        cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - minCost_lastPath);

                cost_aggr_col[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }
            // 重置上一个元素的最小代价数组
            minCost_lastPath = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));

            // 下一个元素
            cost_init_col += direction * width * disp_range;
            cost_aggr_col += direction * width * disp_range;

            last_gray = gray;

        }
    }
}

//左上->右下/右下->左上
void sgm_util::CostAggregateDiagonal1(const cv::Mat &img_data, const sint32 &width, const sint32 &height,
                                      const sint32 &min_disparity, const sint32 &max_disparity, const sint32 &p1,
                                      const sint32 &p2_init, const uint8 *cost_init, uint8 *cost_aggr,
                                      bool is_forward) {

//    assert(width > 1 && height > 1 && max_disparity > min_disparity);
//
//    int count = 0;
//    auto img_data = new uint8[img_data1.cols*img_data1.rows]();
//    for(int i = 0;i < img_data1.rows;i++){
//        for(int j = 0; j < img_data1.cols;j++){
//            img_data[count++] = img_data1.at<uint8_t>(i,j);
//        }
//    }
//
//    // 视差范围
//    const sint32 disp_range = max_disparity - min_disparity;
//
//    // P1,P2
//    const auto& P1 = p1;
//    const auto& P2_Init = p2_init;
//
//    // 正向(左上->右下) ：is_forward = true ; direction = 1
//    // 反向(右下->左上) ：is_forward = false; direction = -1;
//    const sint32 direction = is_forward ? 1 : -1;
//
//    // 聚合
//
//    // 存储当前的行列号，判断是否到达影像边界
//    sint32 current_row = 0;
//    sint32 current_col = 0;
//
//    for (sint32 j = 0; j < width; j++) {
//        // 路径头为每一列的首(尾,dir=-1)行像素
//        auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
//        auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
//        auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);
//
//        // 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
//        std::vector<uint8> cost_last_path(disp_range + 2, UINT8_MAX);
//
//        // 初始化：第一个像素的聚合代价值等于初始代价值
//        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8));
//        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));
//
//        // 路径上当前灰度值和上一个灰度值
//        uint8 gray = *img_col;
//        uint8 gray_last = *img_col;
//
//        // 对角线路径上的下一个像素，中间间隔width+1个像素
//        // 这里要多一个边界处理
//        // 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
//        current_row = is_forward ? 0 : height - 1;
//        current_col = j;
//        if (is_forward && current_col == width - 1 && current_row < height - 1) {
//            // 左上->右下，碰右边界
//            cost_init_col = cost_init + (current_row + direction) * width * disp_range;
//            cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
//            img_col = img_data + (current_row + direction) * width;
//            current_col = 0;
//        }
//        else if (!is_forward && current_col == 0 && current_row > 0) {
//            // 右下->左上，碰左边界
//            cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
//            cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
//            img_col = img_data + (current_row + direction) * width + (width - 1);
//            current_col = width - 1;
//        }
//        else {
//            cost_init_col += direction * (width + 1) * disp_range;
//            cost_aggr_col += direction * (width + 1) * disp_range;
//            img_col += direction * (width + 1);
//        }
//
//        // 路径上上个像素的最小代价值
//        uint8 mincost_last_path = UINT8_MAX;
//        for (auto cost : cost_last_path) {
//            mincost_last_path = std::min(mincost_last_path, cost);
//        }
//
//        // 自方向上第2个像素开始按顺序聚合
//        for (sint32 i = 0; i < height - 1; i ++) {
//            gray = *img_col;
//            uint8 min_cost = UINT8_MAX;
//            for (sint32 d = 0; d < disp_range; d++) {
//                // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
//                const uint8  cost = cost_init_col[d];
//                const uint16 l1 = cost_last_path[d + 1];
//                const uint16 l2 = cost_last_path[d] + P1;
//                const uint16 l3 = cost_last_path[d + 2] + P1;
//                const uint16 l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));
//
//                const uint8 cost_s = cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);
//
//                cost_aggr_col[d] = cost_s;
//                min_cost = std::min(min_cost, cost_s);
//            }
//
//            // 重置上个像素的最小代价值和代价数组
//            mincost_last_path = min_cost;
//            memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));
//
//            // 当前像素的行列号
//            current_row += direction;
//            current_col += direction;
//
//            // 下一个像素,这里要多一个边界处理
//            // 这里要多一个边界处理
//            // 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
//            if (is_forward && current_col == width - 1 && current_row < height - 1) {
//                // 左上->右下，碰右边界
//                cost_init_col = cost_init + (current_row + direction) * width * disp_range;
//                cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
//                img_col = img_data + (current_row + direction) * width;
//                current_col = 0;
//            }
//            else if (!is_forward && current_col == 0 && current_row > 0) {
//                // 右下->左上，碰左边界
//                cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
//                cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
//                img_col = img_data + (current_row + direction) * width + (width - 1);
//                current_col = width - 1;
//            }
//            else {
//                cost_init_col += direction * (width + 1) * disp_range;
//                cost_aggr_col += direction * (width + 1) * disp_range;
//                img_col += direction * (width + 1);
//            }
//
//            // 像素值重新赋值
//            gray_last = gray;
//        }
//    }
//










    //视差范围
    const sint32 disp_range = max_disparity - min_disparity;
    //P1，P2
    const auto &P1 = p1;
    const auto &P2_init = p2_init;
    // 正向(左上->右下) ：is_forward = true ; direction = 1
    // 反向(右下->左上) ：is_forward = false; direction = -1
    const sint32 direction = is_forward ? 1 : -1;

    sint32 current_col = 0;
    sint32 current_row = 0;


    //聚合
    for (sint32 i = 0; i < width; i++) {
        //每一列的首列元素
        auto cost_init_col = (is_forward) ? (cost_init + i * disp_range) : (cost_init +
                                                                            width * disp_range * (height - 1) +
                                                                            i * disp_range);
        auto cost_aggr_col = (is_forward) ? (cost_aggr + i * disp_range) : (cost_aggr +
                                                                            width * disp_range * (height - 1) +
                                                                            i * disp_range);

        // 路径当前的灰度值和上一个灰度值
        uint8 gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);
        uint8 last_gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);

        //上一个元素的代价数组
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        //当前行列
        current_col = i;
        current_row = (is_forward) ? 0 : height - 1;

        //初始化：第一个像素的聚合代价等于初始代价
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));

        //左上->右下 如果到右边界，行继续更新，列回到左边界
        if (direction == 1 && current_col == width - 1) {
            current_row += 1;
            current_col = 0;
        } else if (direction == -1 && current_col == 0) {
            //右下->左上 如果到左边界，行继续更新，列回到右边界
            current_row -= 1;
            current_col = width - 1;
        } else {
            current_col += direction;
            current_row += direction;
        }

        cost_init_col = cost_init + current_row * width * disp_range + current_col * disp_range;
        cost_aggr_col = cost_aggr + current_row * width * disp_range + current_col * disp_range;

        //上一个路径的最小代价
        uint8 minCost_lastPath = UINT8_MAX;
        for (auto cost: cost_last_path) {
            minCost_lastPath = std::min(minCost_lastPath, cost);
        }

        //从第二个元素开始聚合
        for (int j = 1; j < height; j++) {
            gray = img_data.at<uint8_t>(current_row, current_col);
            uint8 min_cost = UINT8_MAX;
            for (int d = 0; d < disp_range; d++) {
                // l1 = L(p-r,d)
                // l2 = L(p-r,d-1) + p1
                // l3 = L(p-r,d+1) + p1
                // l4 = min L(p-r) + p2
                const uint8 cost = cost_init_col[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + P1;
                const uint16 l3 = cost_last_path[d + 2] + P1;
                const uint16 l4 = minCost_lastPath + std::max(P1, P2_init / (abs(gray - last_gray) + 1));

                const uint8 cost_s =
                        cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - minCost_lastPath);

                cost_aggr_col[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }
            // 重置上一个元素的最小代价数组
            minCost_lastPath = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));

            // 下一个元素
            //左上->右下 如果到右边界，行继续更新，列回到左边界
            if (direction == 1 && current_col == width - 1) {
                current_row += direction;
                current_col = 0;
            } else if (direction == -1 && current_col == 0) {
                //右下->左上 如果到左边界，行继续更新，列回到右边界
                current_row += direction;
                current_col = width - 1;
            } else {
                current_col += direction;
                current_row += direction;
            }

            cost_init_col = cost_init + current_row * width * disp_range + current_col * disp_range;
            cost_aggr_col = cost_aggr + current_row * width * disp_range + current_col * disp_range;


        }
    }
}

//右上->左下/左下->右上
void sgm_util::CostAggregateDiagonal2(const cv::Mat &img_data, const sint32 &width, const sint32 &height,
                                      const sint32 &min_disparity, const sint32 &max_disparity, const sint32 &p1,
                                      const sint32 &p2_init, const uint8 *cost_init, uint8 *cost_aggr,
                                      bool is_forward) {
    //视差范围
    const sint32 disp_range = max_disparity - min_disparity;
    //P1，P2
    const auto &P1 = p1;
    const auto &P2_init = p2_init;
    // 正向(右上->左下) ：is_forward = true ; direction = 1
    // 反向(左下->右上) ：is_forward = false; direction = -1
    const sint32 direction = is_forward ? 1 : -1;

    sint32 current_col = 0;
    sint32 current_row = 0;


    //聚合
    for (sint32 i = 0; i < width; i++) {
        //每一列的首列元素
        auto cost_init_col = (is_forward) ? (cost_init + i * disp_range) : (cost_init +
                                                                            width * disp_range *
                                                                            (height - 1) + i * disp_range);
        auto cost_aggr_col = (is_forward) ? (cost_aggr + i * disp_range) : (cost_aggr +
                                                                            width * disp_range *
                                                                            (height - 1) + i * disp_range);

        // 路径当前的灰度值和上一个灰度值
        uint8 gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);
        uint8 last_gray = (is_forward) ? img_data.at<uint8_t>(0, i) : img_data.at<uint8_t>(height - 1, i);

        //上一个元素的代价数组
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        //当前行列
        current_col = i;
        current_row = (is_forward) ? 0 : (height - 1);

        //初始化：第一个像素的聚合代价等于初始代价
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));

        //右上->左下 如果到左边界，行继续更新，列回到右边界
        if (direction == 1 && current_col == 0) {
            current_row += 1;
            current_col = width - 1;
        } else if (direction == -1 && current_col == width - 1) {
            //左下->右上 如果到右边界，行继续更新，列回到左边界
            current_row -= 1;
            current_col = 0;
        } else {
            if (is_forward) {
                current_col -= 1;
                current_row += 1;
            } else {
                current_col += 1;
                current_row -= 1;
            }
        }

        cost_init_col = cost_init + current_row * width * disp_range + current_col * disp_range;
        cost_aggr_col = cost_aggr + current_row * width * disp_range + current_col * disp_range;

        //上一个路径的最小代价
        uint8 minCost_lastPath = UINT8_MAX;
        for (auto cost: cost_last_path) {
            minCost_lastPath = std::min(minCost_lastPath, cost);
        }

        //从第二个元素开始聚合
        for (int j = 1; j < height; j++) {
            gray = img_data.at<uint8_t>(current_row, current_col);
            uint8 min_cost = UINT8_MAX;
            for (int d = 0; d < disp_range; d++) {
                // l1 = L(p-r,d)
                // l2 = L(p-r,d-1) + p1
                // l3 = L(p-r,d+1) + p1
                // l4 = min L(p-r) + p2
                const uint8 cost = cost_init_col[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + P1;
                const uint16 l3 = cost_last_path[d + 2] + P1;
                const uint16 l4 = minCost_lastPath + std::max(P1, P2_init / (abs(gray - last_gray) + 1));

                const uint8 cost_s =
                        cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - minCost_lastPath);

                cost_aggr_col[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }
            // 重置上一个元素的最小代价数组
            minCost_lastPath = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));

            // 下一个元素
            //右上->左下 如果到左边界，行继续更新，列回到右边界
            if (direction == 1 && current_col == 0) {
                current_row += 1;
                current_col = width - 1;
            } else if (direction == -1 && current_col == width - 1) {
                //左下->右上 如果到右边界，行继续更新，列回到左边界
                current_row -= 1;
                current_col = 0;
            } else {
                if (is_forward) {
                    current_col -= 1;
                    current_row += 1;
                } else {
                    current_col += 1;
                    current_row -= 1;
                }
            }

            cost_init_col = cost_init + current_row * width * disp_range + current_col * disp_range;
            cost_aggr_col = cost_aggr + current_row * width * disp_range + current_col * disp_range;
            last_gray = gray;

        }
    }
}

void sgm_util::RemoveSpeckles(float32 *disparity_map, const uint32 &width, const uint32 &height,
                              const uint32 &diff_insame, const uint32 &min_speckle_area) {
//    assert(width > 0 && height > 0);
//    if (width < 0 || height < 0) {
//        return;
//    }
//
//    // 定义标记像素是否访问的数组
//    std::vector<bool> visited(uint32(width*height),false);
//    for(sint32 i=0;i<height;i++) {
//        for(sint32 j=0;j<width;j++) {
//            if (visited[i * width + j] || disparity_map[i*width+j] == Invalid_Float) {
//                // 跳过已访问的像素及无效像素
//                continue;
//            }
//            // 广度优先遍历，区域跟踪
//            // 把连通域面积小于阈值的区域视差全设为无效值
//            std::vector<std::pair<sint32, sint32>> vec;
//            vec.emplace_back(i, j);
//            visited[i * width + j] = true;
//            uint32 cur = 0;
//            uint32 next = 0;
//            do {
//                // 广度优先遍历区域跟踪
//                next = vec.size();
//                for (uint32 k = cur; k < next; k++) {
//                    const auto& pixel = vec[k];
//                    const sint32 row = pixel.first;
//                    const sint32 col = pixel.second;
//                    const auto& disp_base = disparity_map[row * width + col];
//                    // 8邻域遍历
//                    for(int r=-1;r<=1;r++) {
//                        for(int c=-1;c<=1;c++) {
//                            if(r==0&&c==0) {
//                                continue;
//                            }
//                            int rowr = row + r;
//                            int colc = col + c;
//                            if (rowr >= 0 && rowr < height && colc >= 0 && colc < width) {
//                                if(!visited[rowr * width + colc] &&
//                                   (disparity_map[rowr * width + colc] != Invalid_Float) &&
//                                   abs(disparity_map[rowr * width + colc] - disp_base) <= diff_insame) {
//                                    vec.emplace_back(rowr, colc);
//                                    visited[rowr * width + colc] = true;
//                                }
//                            }
//                        }
//                    }
//                }
//                cur = next;
//            } while (next < vec.size());
//
//            // 把连通域面积小于阈值的区域视差全设为无效值
//            if(vec.size() < min_speckle_area) {
//                for(auto& pix:vec) {
//                    disparity_map[pix.first * width + pix.second] = Invalid_Float;
//                }
//            }
//        }
//    }

    assert(height > 0 && width > 0);

    std::vector<bool> visited((width * height), false);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (visited[i * width + j] || disparity_map[i * width + j] == Invalid_Float) {
                continue;
            }
            std::vector<std::pair<sint32, sint32>> pair_vec;
            pair_vec.emplace_back(i, j);
            visited[i * width + j] = true;
            //广度优先遍历
            uint32 cur = 0;
            uint32 next = 0;
            do {
                next = pair_vec.size();
                for (uint32 k = cur; k < next; k++) {
                    auto pair = pair_vec[k];
                    sint32 row = pair.first;
                    sint32 col = pair.second;
                    auto base_disp = disparity_map[row * width + col];
                    //8邻域遍历
                    for (sint32 r = -1; r <= 1; r++) {
                        for (sint32 c = -1; c <= 1; c++) {
                            if (r == 0 && c == 0) {
                                //中心区域
                                continue;
                            }
                            sint32 row_r = row + r;
                            sint32 col_c = col + c;
                            if (row_r >= 0 && row_r < height && col_c >= 0 && col_c < width &&
                                abs(base_disp - disparity_map[row_r * width + col_c]) <= float(diff_insame) &&
                                disparity_map[row_r * width + col_c] != Invalid_Float &&
                                !visited[row_r * width + col_c]) {
                                //(row,col是连通的)
                                pair_vec.emplace_back(row_r, col_c);
                                visited[row_r * width + col_c] = true;
                            }
                        }
                    }
                }
                cur = next;
            } while (next < pair_vec.size());
            if (pair_vec.size() < min_speckle_area) {
                for (auto &e: pair_vec) {
                    disparity_map[e.first * width + e.second] = Invalid_Float;
                }
            }
        }
    }
}