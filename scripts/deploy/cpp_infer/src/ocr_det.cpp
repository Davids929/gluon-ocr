
#include "include/ocr_det.hpp"

void DBDetector::Run(cv::Mat &image, std::vector<std::vector<std::vector<int>>> &boxes){
      // set input and bind executor
      auto data = AsData(image, ctx_);
      auto input_shape = data.GetShape();
      this->args_map_["data"] = data;
      this->exec_ = net_.SimpleBind(
        this->ctx_, this->args_map_, std::map<std::string, NDArray>(),
        std::map<std::string, OpReqType>(), this->auxs_map_);
      // begin forward
      NDArray::WaitAll();
      auto start = std::chrono::steady_clock::now();
      this->exec_->Forward(false);
      auto pred = this->exec_->outputs[0].Copy(Context(kCPU, 0));
      NDArray::WaitAll();
      auto end = std::chrono::steady_clock::now();
      int h = input_shape[2];
      int w = input_shape[3];
      int n = h * w;
      std::vector<float> out_data(n, 0.0);
      pred.SyncCopyToCPU(&out_data, n);

      cv::Mat pred_map(h, w, CV_32F, (float *)out_data.data());

      cv::Mat bina_map;
      cv::threshold(pred_map, bina_map, this->thresh_, 1, cv::THRESH_BINARY);
      bina_map.convertTo(bina_map, CV_8UC1);
      boxes = this->post_process_.BoxesFromBitmap(
            pred_map, bina_map, this->box_thresh_, this->unclip_ratio_);

}

int main(){
  std::cout<<"Hello gluon-ocr!"<<std::endl;
  DBDetector det("/home/sw/demo/receipt_recognition/model/data/resnet50-db-symbol.json", 
                     "/home/sw/demo/receipt_recognition/model/data/resnet50-db-0000.params", 
                     1, 2048, 768, 0.3, 0.3, 2.0, true);
    
  cv::Mat image = cv::imread("/home/sw/demo/receipt_recognition/test_imgs/20200821165522.jpg", 1);
  std::vector<std::vector<std::vector<int>>> boxes;
  det.Run(image, boxes);
}