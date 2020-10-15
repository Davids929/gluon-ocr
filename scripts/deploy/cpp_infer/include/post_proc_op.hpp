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

        std::vector<std::vector<int>>
        OrderPointsClockwise(std::vector<std::vector<int>> pts);

    private:
        const int min_size_ = 3;
        const int max_candidates_ = 1000;
};

template<class T> bool Xsort(std::vector<T> a, std::vector<T> b){
    if (a[0] != b[0])
        return a[0] < b[0];
    return false;
};

template<class T> T clamp(T x, T min, T max){
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

