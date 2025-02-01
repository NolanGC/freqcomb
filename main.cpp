#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "mergeFrequencies.hpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input1> <input2> <mask> <output_base>\n";
        std::cerr << "Example: ./program image1.jpg image2.jpg mask.jpg results\n";
        return 1;
    }

    // Load images
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    cv::Mat mask = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    
    std::cout << img1.channels() << std::endl;
    
    if (img1.empty() || img2.empty() || mask.empty()) {
        std::cerr << "Error loading images!\n";
        return 1;
    }

    //cv::Mat merged = mergeFrequencies(img1, img2,
    //                                51, 15.0,
    //                                15, 3.0,
    //                                0.65f);
    float radius = 10.0f;
    float sigma = radius;
    int kernelSize = static_cast<int>(6 * sigma + 1);
    if ((kernelSize & 1) == 0) kernelSize++;
    float blend_strength = 0.65;
    cv::Mat merged = mergeFrequencies(img1, img2, mask, kernelSize, sigma, blend_strength);
    
    std::string filename = argv[4];
    if (!cv::imwrite(filename, merged)) {
        std::cerr << "Failed to save: " << filename << "\n";
    }
    else {
        std::cout << "Saved: " << filename << "\n";
    }
}
