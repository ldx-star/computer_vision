//
// Created by liangdaxin on 23-6-19.
//

#include "sgm_util.h"

void sgm_util::census_transform_5x5(const cv::Mat &img, cv::Mat& census, const sint32 &width, const sint32 &height) {
    census = cv::Mat::zeros(img.size(),CV_8U);
    for(int i = 2; i < height - 2; i++ ){
        for(int j = 2; j < width - 2; j++){
            const uint8 center = img.at<u_int8_t>(i,j);
            uint8 census_value = 0;
            //遍历center的邻域
            for(int r = -2 ; r < 2; r++){
                for(int c = -2; c < 2; c++){
                    if(img.at<u_int8_t>(i + r, j + c) < center){
                        census_value++;
                    }
                }
            }
            census.at<u_int8_t>(i,j) = census_value;
        }
    }
}