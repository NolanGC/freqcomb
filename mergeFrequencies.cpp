#include "mergeFrequencies.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat imageBlend(const cv::Mat& img_a, const cv::Mat& img_b, float blend_percentage) {
    cv::Mat result;
    cv::addWeighted(img_a, 1-blend_percentage, img_b, blend_percentage, 0.0, result);
    return result;
}

cv::Mat maskedImageBlend(const cv::Mat& original, const cv::Mat& background, const cv::Mat& mask) {
    cv::Mat inv_mask;
    cv::bitwise_not(mask, inv_mask);
    cv::Mat original_float;
    original.convertTo(original_float, CV_64F, 1.0/255.0);
    cv::Mat background_float;
    background.convertTo(background_float, CV_64F, 1.0/255.0);
    cv::Mat mask_float;
    mask.convertTo(mask_float, CV_64F, 1.0/255.0);
    inv_mask.convertTo(inv_mask, CV_64F, 1.0/255.0);
    cv::Mat img1, img2;
    cv::multiply(original_float, mask_float, img1);
    cv::multiply(background_float, inv_mask, img2);
    cv::add(img1, img2, img1);
    img1.convertTo(img1, CV_8U, 255.0);
    return img1;
}

cv::Mat adjustLevels(cv::Mat image, double black_level, double mid_level, double white_level) {
    cv::Mat img_float;
    image.convertTo(img_float, CV_32FC3, 1.0/255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);
    const double black = black_level/255.0;
    const double white = white_level/255.0;
    const double scale = 1.0/(white - black);
    
    for(auto& channel : channels) {
        cv::threshold(channel, channel, black, 1.0, cv::THRESH_TOZERO);
        channel = (channel - black) * scale;
        cv::Mat lower = cv::Mat::zeros(channel.size(), CV_32FC1);
        cv::Mat upper = cv::Mat::ones(channel.size(), CV_32FC1);
        cv::max(channel, lower, channel);
        cv::min(channel, upper, channel);
        const double mid = (mid_level/255.0 - black)/ (white - black);
        const double gamma = mid > 0 ? std::log(0.5)/std::log(mid) : 1.0;
        cv::pow(channel, std::max(-10.0, std::min(gamma, 10.0)), channel);
    }
    
    cv::merge(channels, img_float);
    img_float.convertTo(img_float, CV_8UC3, 255.0);
    return img_float;
}

cv::Mat mergeFrequencies(const cv::Mat& imposed_img, const cv::Mat& relit_img,
                       const cv::Mat& mask, int kernel_size, float sigma,
                       float blend_strength) {
    CV_Assert(!imposed_img.empty() && !relit_img.empty() && !mask.empty());
    cv::Mat imposed_img_up;
    cv::resize(imposed_img, imposed_img_up, {}, 2.0, 2.0, cv::INTER_CUBIC);
    
    cv::Mat relit_img_resized;
    if(imposed_img_up.size() != relit_img.size()) {
        cv::resize(relit_img, relit_img_resized, imposed_img_up.size());
    } else {
        relit_img.copyTo(relit_img_resized);
    }
    
    cv::Mat mask_up;
    cv::resize(mask, mask_up, imposed_img_up.size(), 0, 0, cv::INTER_CUBIC);
    
    cv::Mat inverted_imposed_img, lowpass_imposed_img;
    cv::bitwise_not(imposed_img_up, inverted_imposed_img);
    cv::GaussianBlur(imposed_img_up, lowpass_imposed_img, {kernel_size,kernel_size}, sigma, sigma);
    
    cv::Mat merged_imposed_img = imageBlend(inverted_imposed_img, lowpass_imposed_img, 0.5f);
    cv::bitwise_not(merged_imposed_img, merged_imposed_img);
    cv::Mat processed_imposed_img = imageBlend(lowpass_imposed_img, merged_imposed_img, 1.0f);
    
    cv::Mat inverted_relit_img, lowpass_relit_img;
    cv::bitwise_not(relit_img_resized, inverted_relit_img);
    cv::GaussianBlur(relit_img_resized, lowpass_relit_img, {kernel_size,kernel_size}, sigma, sigma);
    
    cv::Mat merged_relit_img = imageBlend(inverted_relit_img, lowpass_relit_img, 0.5f);
    cv::bitwise_not(merged_relit_img, merged_relit_img);
    cv::Mat processed_relit_img = imageBlend(lowpass_relit_img, merged_relit_img, 1.0f);
    
    cv::Mat combined_img = maskedImageBlend(processed_imposed_img, processed_relit_img, mask_up);
    cv::Mat result;
    cv::resize(imageBlend(lowpass_relit_img, combined_img, 0.65f), result, {}, 0.5, 0.5, cv::INTER_AREA);
    return adjustLevels(result, 55.0, 130.0, 165.0);
}
