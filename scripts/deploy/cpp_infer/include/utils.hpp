#pragma once

#include <iomanip>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


// string split
std::vector<std::string> Split(std::string &str, const std::string delim);

// string Delect
std::string Delete(std::string &str, const std::string delim);


std::vector<std::string> ReadFile(const std::string dict_path);

std::map<std::string, std::string> LoadConfig(const std::string &config_path);

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

std::vector<std::vector<int>>
OrderPointsClockwise(std::vector<std::vector<int>> pts);

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