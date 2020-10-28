
#include "include/ocr_det.hpp"

void DBDetector::Run(cv::Mat &image, std::vector<std::vector<std::vector<int>>> &boxes){
      // set input and bind executor
      cv::Mat resized_img = ResizeShortWithin(image, this->min_side_len_, this->max_side_len_, 32, false);
      auto data = AsData(resized_img, this->ctx_);
      auto input_shape = data.GetShape();
      this->args_map_["data"] = data;
      this->exec_ = net_.SimpleBind(
        this->ctx_, this->args_map_, std::map<std::string, NDArray>(),
        std::map<std::string, OpReqType>(), this->auxs_map_);
      // begin forward
      //NDArray::WaitAll();
      auto start = std::chrono::steady_clock::now();
      this->exec_->Forward(false);
      auto pred = this->exec_->outputs[0].Copy(Context(kCPU, 0));
      //NDArray::WaitAll();
      auto end = std::chrono::steady_clock::now();
      pred.WaitToRead();
      int h = input_shape[2];
      int w = input_shape[3];
      int n = h * w;
      pred = pred.Reshape(Shape(n));
      
      std::vector<float> out_data;
      pred.SyncCopyToCPU(&out_data, n);
      cv::Mat pred_map(h, w, CV_32F, out_data.data());
      
      cv::Mat bina_map, pred_img;
      pred_img = pred_map*255;
      pred_img.convertTo(pred_img, CV_8UC1);
      cv::imwrite("./pred_map.jpg", pred_img);
      double thresh = double(this->thresh_*255);
      cv::threshold(pred_img, bina_map, thresh, 255, cv::THRESH_BINARY);
      
      //cv::imwrite("./bina_map.jpg", bina_map);
      boxes = this->post_process_.BoxesFromBitmap(
            pred_map, bina_map, this->box_thresh_, this->unclip_ratio_);
      int origin_h = image.rows;
      int origin_w = image.cols;
      //std::cout<< "boxes number :"<<boxes.size()<<std::endl;
      GetOriginScaleBox(boxes, origin_h, origin_w, h, w);
}

void DBDetector::GetOriginScaleBox(std::vector<std::vector<std::vector<int>>> &boxes, 
                               int &origin_h, int &origin_w, int &img_h, int &img_w){
    
    float ratio_h = float(origin_h)/float(img_h);
    float ratio_w = float(origin_w)/float(img_w);
    for (int i=0; i<boxes.size(); i++){
        boxes[i] = OrderPointsClockwise(boxes[i]);
        for (int j=0; j<boxes[0].size(); j++){
           int x = int(float(boxes[i][j][0])*ratio_w);
           int y = int(float(boxes[i][j][1])*ratio_h);
           boxes[i][j][0] = clamp(x, 0, origin_w);
           boxes[i][j][1] = clamp(y, 0, origin_h);
        }
    }
}

// int main(){
//   std::cout<<"Hello gluon-ocr!"<<std::endl;
//   DBDetector det("/home/sw/demo/receipt_recognition/model/data/resnet50-db-symbol.json", 
//                      "/home/sw/demo/receipt_recognition/model/data/resnet50-db-0000.params", 
//                      1, 2048, 768, 0.3, 0.3, 2.0, true);
    
//   cv::Mat image = cv::imread("/home/sw/demo/receipt_recognition/test_imgs/20200821165522.jpg", 1);
//   std::vector<std::vector<std::vector<int>>> boxes;
//   det.Run(image, boxes);
// }