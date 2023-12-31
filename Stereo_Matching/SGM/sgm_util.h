//
// Created by liangdaxin on 23-6-19.
//

#ifndef SGM_SGM_UTIL_H
#define SGM_SGM_UTIL_H
#include <opencv2/opencv.hpp>
#include "sgm_types.h"
#include <string>

namespace sgm_util {
    void census_transform_5x5(const cv::Mat& img, cv::Mat& census, const sint32& width, const sint32& height);
    uint32 Hamming32(const uint32& x,const uint32& y);

    /*
     * @brief 左右路径聚合
     *
     * @param img_data          图像数据
     * @param width             图像宽
     * @param height            图像高
     * @param min_disparity     最小视差
     * @param max_disparity     最大视差
     * @param p1                惩罚项p1
     * @param p2_init           惩罚项p2_init
     * @param cost_init         初始代价
     * @param cost_aggr         聚合代价
     * @param is_forward        是否为正方向（从左到右为正，从右到左为反）
     */
    void CostAggregateLeftRight(const cv::Mat& img_data, const sint32& width,const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
                                const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);


    void CostAggregateUpDown(const cv::Mat& img_data, const sint32& width,const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
                                const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);

    //左上->右下/右下->左上
    void CostAggregateDiagonal1(const cv::Mat& img_data, const sint32& width,const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
                             const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);

    //右上->左下/左下->右上
    void CostAggregateDiagonal2(const cv::Mat& img_data, const sint32& width,const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
                                const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);

    void save_img(cv::Mat& img,const std::string& filename);

    /*
     *  @brief 移除小连通区域
     *  @param disparity_map    视差图
     *  @param width            宽
     *  @param height           高
     *  @param diff_insame      同一连通区域内的局部像素差异
     *  @param min_spackle_area 最小连通区面积
     *  @param invalid_val      无效值
     */
    void RemoveSpeckles(float32* disparity_map, const uint32& width, const uint32& height,const uint32& diff_insame, const uint32& min_speckle_area);

    /*
     *  @brief 中值滤波
     *  @param in           数据源
     *  @param out          目标数据
     *  @param width        宽度
     *  @param height       高度
     *  @param window_size  窗口宽度
     */
    void MedianFilter(const float32* in,float32* out, const uint32& width, const uint32& height, const sint32& window_size);
};


#endif //SGM_SGM_UTIL_H
