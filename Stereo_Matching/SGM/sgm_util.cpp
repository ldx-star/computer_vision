//
// Created by liangdaxin on 23-6-19.
//

#include "sgm_util.h"

void sgm_util::census_transform_5x5(const cv::Mat &img, cv::Mat& census, const sint32 &width, const sint32 &height) {
    census = cv::Mat::zeros(img.size(),CV_32S);
    for(int i = 2; i < height - 2; i++ ){
        for(int j = 2; j < width - 2; j++){
            const uint8 center = img.at<u_int8_t>(i,j);
            sint32 census_value = 0;
            //遍历center的邻域
            for(int r = -2 ; r < 2; r++){
                for(int c = -2; c < 2; c++){
                    census_value <<= 1;
                    if(img.at<u_int8_t>(i + r, j + c) < center){
                        census_value += 1;
                    }
                }
            }
            census.at<u_int32_t>(i,j) = census_value;
        }
    }
}

uint32 sgm_util::Hamming32(const uint32 &x, const uint32 &y) {
    uint32 distance = 0;
    uint32 val = x ^ y;
    //统计val中1个数
    while(val){
        if(val&1){
            distance++;
        }
        val >>= 1;
    }
    return distance;
}