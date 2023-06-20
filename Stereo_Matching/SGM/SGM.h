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

        SGMOption(): num_paths(8), min_disparity(0),max_disparity(64),censusSize(Census5x5),p1(10),p2(150){}
    };

    bool Initialize(const uint32& width, const uint32& height,const SGMOption& option);
    bool Match(const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& disp_left);
    bool Reset(const uint32& width, const uint32& height, const SGMOption& option);
private:
    void CensusTransform();
    void ComputeCost();
    void ComputeDisparity();
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
};


#endif //SGM_SGM_H
