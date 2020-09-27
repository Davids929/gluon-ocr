
#include "include/ocr_det.hpp"


void DBDetector::Run(cv::Mat &image, std::vector<std::vector<std::vector<int>>> &boxes){
    // set input and bind executor
    auto data = AsData(image, ctx_);
    args_["data"] = data;
    Executor *exec = net_.SimpleBind(
      ctx, args_, std::map<std::string, NDArray>(),
      std::map<std::string, OpReqType>(), auxs_);
    // begin forward
    NDArray::WaitAll();
    auto start = std::chrono::steady_clock::now();
    exec->Forward(false);
    auto pred = exec->outputs[0].Copy(Context(kCPU, 0));
    NDArray::WaitAll();
    auto end = std::chrono::steady_clock::now();


}