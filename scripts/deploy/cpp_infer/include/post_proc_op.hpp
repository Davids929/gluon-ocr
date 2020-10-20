#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include "include/clipper.hpp"
#include "include/common.hpp"
#include "include/utils.hpp"

class DBPostProcess{
    public:

        std::vector<std::vector<std::vector<int>>>
        BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                        const float &box_thresh, const float &unclip_ratio); 

        cv::RotatedRect UnClip(std::vector<std::vector<float>> box,
                         const float &unclip_ratio);

        void GetContourArea(const std::vector<std::vector<float>> &box,
                      float unclip_ratio, float &distance);

        std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box);

        float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);


    private:
        const int min_size_ = 3;
        const int max_candidates_ = 1000;
};

