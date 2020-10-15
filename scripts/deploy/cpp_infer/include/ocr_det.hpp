
#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "mxnet-cpp/MxNetCpp.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include "include/common.hpp"
#include "include/post_proc_op.hpp"

using namespace mxnet::cpp;

class DBDetector{
    public:
        explicit DBDetector(const std::string &model_path, 
                            const std::string &params_path,
                            const int &gpu_id, 
                            const int &max_side_len,
                            const int &min_side_len,
                            const double &thresh,
                            const double &box_thresh,
                            const double &unclip_ratio,
                            const bool &visualize){

            this->min_side_len_ = min_side_len; 
            this->max_side_len_ = max_side_len;
            this->thresh_ = thresh;
            this->box_thresh_ = box_thresh;
            this->unclip_ratio_ = unclip_ratio;
            this->visualize_ = visualize;

            if (gpu_id >= 0){
                this->ctx_ = Context::gpu(gpu_id);
            }

            LoadCheckpoint(model_path, params_path, &net_, &args_map_, &auxs_map_, this->ctx_);

        };

        void Run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes);
        
        ~DBDetector(){
            delete this->exec_;
        }

    public:
        Symbol net_;
    
    private:
        int min_side_len_ = 736;
        int max_side_len_ = 2048;
        double thresh_ = 0.3;
        double box_thresh_ = 0.5;
        double unclip_ratio_ = 2.0;
        bool visualize_ = true;
        Context ctx_ = Context::cpu();
        std::map<std::string, NDArray> args_map_, auxs_map_;
        Executor *exec_;
        DBPostProcess post_process_;

};