#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "mxnet-cpp/MxNetCpp.h"
#include "include/common.hpp"
#include "include/utils.hpp"

using namespace mxnet::cpp;

class CRNNRecognizer{
    public:
        CRNNRecognizer(const std::string &model_path, 
                       const std::string &params_path,
                       const std::string &dict_path,
                       const int &gpu_id,
                       const int max_side_len,
                       const int short_side,
                       const int num_buckets){

            if (gpu_id >= 0)
                this->ctx_  = Context::gpu(gpu_id);

            this->max_side_len_ = max_side_len;
            this->short_side_   = short_side;
            this->step_         = max_side_len / num_buckets;
            for (int i=0; i<num_buckets; i++){
                int bucket = this->step_ * (i+1);
                if (i == num_buckets-1)
                    this->bucket_keys_.push_back(max_side_len);
                else
                    this->bucket_keys_.push_back(bucket);
            }

            this->voc_dict_ = ReadFile(dict_path);
            this->voc_size_ = this->voc_dict_.size(); 
            LoadCheckpoint(model_path, params_path, &net_, &args_map_, &auxs_map_, ctx_);
            InitModel();
        }

        ~CRNNRecognizer(){
            for (auto bucket : this->exec_buckets_) {
                Executor* executor = bucket.second;
                delete executor;
            }
        }

    public:
        Symbol net_;
        void InitModel();
        std::string Run(cv::Mat &image);
        void Run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes,
                 std::vector<std::string> &texts); 

    private:
        int max_side_len_;
        int short_side_;
        int voc_size_;
        int step_;
        Context ctx_ = Context::cpu();
        std::map<std::string, NDArray> args_map_, auxs_map_;
        std::vector<int> bucket_keys_;
        std::map<int, Executor*> exec_buckets_;
        std::vector<std::string> voc_dict_;

    private:
        cv::Mat GetRotateCropImage(const cv::Mat &image, 
                                   std::vector<std::vector<int>> box);
        int GetBucketKey(int img_w);
        std::string PostProcess(std::vector<float> &pred_id, int blank);
};