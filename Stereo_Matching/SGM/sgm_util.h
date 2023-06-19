//
// Created by liangdaxin on 23-6-19.
//

#ifndef SGM_SGM_UTIL_H
#define SGM_SGM_UTIL_H
#include <opencv2/opencv.hpp>
#include "sgm_types.h"

namespace sgm_util {
    void census_transform_5x5(const cv::Mat& img, cv::Mat& census, const sint32& width, const sint32& height);
};


#endif //SGM_SGM_UTIL_H
