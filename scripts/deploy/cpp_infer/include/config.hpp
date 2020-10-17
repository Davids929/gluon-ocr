
#pragma once

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "include/utils.hpp"

class Config {

public:
  Config(const std::string config_path){
    auto config_map = LoadConfig(config_path);
    
    this->gpu_id = std::stoi(config_map["gpu_id"]);
    this->db_max_side_len = std::stoi(config_map["db_max_side_len"]);
    this->db_min_side_len = std::stoi(config_map["db_min_side_len"]);
    this->db_thresh = std::stoi(config_map["db_thresh"]);
    this->db_box_thresh = std::stoi(config_map["db_box_thresh"]);
    this->db_unclip_ratio = std::stoi(config_map["db_unclip_ratio"]);
    this->det_model_path  = config_map["det_model_path"];
    this->det_params_path = config_map["det_params_path"];
    
    this->rec_max_side_len = std::stoi(config_map["rec_max_side_len"]);
    this->rec_short_side = std::stoi(config_map["rec_short_side"]);
    this->num_buckets = std::stoi(config_map["num_buckets"]);
    this->rec_model_path  = config_map["rec_model_path"];
    this->rec_params_path = config_map["rec_params_path"];
    this->voc_dict_path   = config_map["voc_dict_path"];
  }

public:
  int gpu_id = 0;
  //detect model
  int db_max_side_len = 1440;
  int db_min_side_len = 768;
  double db_thresh = 0.3;
  double db_box_thresh = 0.4;
  double db_unclip_ratio = 2.0;
  std::string det_model_path;
  std::string det_params_path;

  //recog model
  int rec_max_side_len = 1024;
  int rec_short_side   = 32;
  int num_buckets      = 1;
  std::string voc_dict_path;
  std::string rec_model_path;
  std::string rec_params_path;
  
  //
  bool visualize = false;
      
};


