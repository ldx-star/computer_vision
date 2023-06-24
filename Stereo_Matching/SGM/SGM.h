//
// Created by liangdaxin on 23-6-19.
//

#ifndef SGM_SGM_H
#define SGM_SGM_H
#include "sgm_types.h"
#include "sgm_util.h"
#include <opencv2/opencv.hpp>


class SGM {
public:
    SGM();
    ~SGM();

    enum CensusSize{
        Census5x5 = 0
    };

    struct SGMOption{
        uint8 num_paths; //  聚合路径
        sint32 min_disparity; // 最小视差
        sint32 max_disparity; // 最大视差


        sint32 p1; // 惩罚项参数 p1
        sint32 p2; // 惩罚项参数 p2


        CensusSize censusSize;

        SGMOption(): num_paths(4), min_disparity(0),max_disparity(64),censusSize(Census5x5),p1(10),p2(150){}
    };

    bool Initialize(const uint32& width, const uint32& height,const SGMOption& option);
    bool Match(const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& disp_left);
    bool Reset(const uint32& width, const uint32& height, const SGMOption& option);
private:
    void CensusTransform();
    //代价计算
    void ComputeCost();
    //代价聚合
    void CostAggregation();
    //视差计算
    void ComputeDisparity(uint8* cost_ptr);
private:
    bool is_initialized_; // 是否初始化标志
    cv::Mat left_img_;
    cv::Mat right_img_;
    uint32 width_;
    uint32 height_;
    SGMOption option_;
    cv::Mat census_left_;
    cv::Mat census_right_;
    uint8* cost_init_; // （rows,cols,disp_range）
    cv::Mat disp_left_;

     uint8* cost_aggr_;// 聚合匹配代价

    uint8* cost_aggr_1_;// 聚合匹配代价-方向1  左 -> 右
    uint8* cost_aggr_2_;// 聚合匹配代价-方向2  右 -> 左
    uint8* cost_aggr_3_;// 聚合匹配代价-方向3  上 -> 下
    uint8* cost_aggr_4_;// 聚合匹配代价-方向4  下 -> 上
    uint8* cost_aggr_5_;// 聚合匹配代价-方向5  左上 -> 右下
    uint8* cost_aggr_6_;// 聚合匹配代价-方向6  右下 -> 左上
    uint8* cost_aggr_7_;// 聚合匹配代价-方向7  右上 -> 左下
    uint8* cost_aggr_8_;// 聚合匹配代价-方向8  左下 -> 右上
};


#endif //SGM_SGM_H
