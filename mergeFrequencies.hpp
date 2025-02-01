//
//  mergeFrequencies.hpp
//  learning
//
//  Created by Nolan Clement on 1/30/25.
//

#ifndef mergeFrequencies_hpp
#define mergeFrequencies_hpp

#include <stdio.h>
#include <opencv2/core.hpp>

cv::Mat mergeFrequencies(const cv::Mat& imposed_img, const cv::Mat& relit_img,
                          const cv::Mat& mask, int kernel_size, float sigma,
                          float blend_strength);

#endif /* mergeFrequencies_hpp */
