
#include "include/ocr_rec.hpp"

void CRNNRecognizer::InitModel(){

    std::vector<NDArray> arg_arrays;
    std::vector<NDArray> grad_arrays;
    std::vector<OpReqType> grad_reqs;
    std::vector<NDArray> aux_arrays;

    int highest_bucket_key = this->bucket_keys_.back();
    this->args_map_["data"] = NDArray(Shape(1, 3, this->short_side_, highest_bucket_key), 
                                  this->ctx_, false);
    this->net_.InferExecutorArrays(this->ctx_, &arg_arrays, &grad_arrays, 
                                   &grad_reqs, &aux_arrays, this->args_map_,
                                   std::map<std::string, NDArray>(),
                                   std::map<std::string, OpReqType>(),
                                   this->auxs_map_);
    Executor *master_executor = this->net_.Bind(this->ctx_, arg_arrays, grad_arrays, 
                                                grad_reqs, aux_arrays,
                                                std::map<std::string, Context>(), nullptr);
    
    this->exec_buckets_[highest_bucket_key] = master_executor;
    for (int bucket : this->bucket_keys_){
        if (this->exec_buckets_.find(bucket) == this->exec_buckets_.end()) {
            arg_arrays[0]  = NDArray(Shape(1, 3, this->short_side_, bucket), this->ctx_, false);
            Executor *executor = this->net_.Bind(this->ctx_, arg_arrays, grad_arrays, 
                                                 grad_reqs, aux_arrays,
                                                 std::map<std::string, Context>(), master_executor);
            exec_buckets_[bucket] = executor;
        }
    }
};

std::string CRNNRecognizer::Run(cv::Mat &image){
    
    cv::Mat resize_img = ResizeShortWithin(image, this->short_side_, this->max_side_len_, 8, true);
    int img_h = resize_img.rows;
    int img_w = resize_img.cols;
    int bucket_key = GetBucketKey(img_w);
    if (img_w < bucket_key){
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, bucket_key - img_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
    NDArray data = AsData(resize_img, this->ctx_);
    std::cout<<"current bucket key:"<<bucket_key<<std::endl;
    Executor* exec = this->exec_buckets_[bucket_key];
    data.CopyTo(&(exec->arg_dict()["data"]));
    // begin forward
    NDArray::WaitAll();
    auto start = std::chrono::steady_clock::now();
    exec->Forward(false);
    NDArray output = exec->outputs[0].Copy(Context(kCPU, 0));
    NDArray::WaitAll();
    auto end = std::chrono::steady_clock::now();

    NDArray pred;
    Operator("argmax")(output, 2).Invoke(pred);
    std::vector<unsigned int> out_shape = pred.GetShape();
    //std::cout<<"out dim:"<<out_shape[1]<< "voc size:"<<this->voc_size_<<std::endl;
    std::string text = "123";//PostProcess(pred, this->voc_size_);
    return text;
}

void CRNNRecognizer::Run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes, 
                         std::vector<std::string> &texts){
    cv::Mat crop_img;
    for (int i=0; i<boxes.size(); i++){
        auto t1 = std::chrono::system_clock::now();
        crop_img = GetRotateCropImage(img, boxes[i]);
        auto t2 = std::chrono::system_clock::now();
        std::string text = Run(crop_img);
        auto t3 = std::chrono::system_clock::now();
        texts.push_back(text);
        std::cout<<"crop image cost time:"<<std::chrono::duration<double, std::milli>(t2 - t1).count()/1000<<" s";
        std::cout<<" recog cost time:"<<std::chrono::duration<double, std::milli>(t3 - t2).count()/1000<<" s"<<std::endl;
    }    
}

cv::Mat CRNNRecognizer::GetRotateCropImage(const cv::Mat &image,
                                           std::vector<std::vector<int>> box){
    
    int crop_width = int(sqrt(pow(box[0][0] - box[1][0], 2) +
                              pow(box[0][1] - box[1][1], 2)));
    int crop_height = int(sqrt(pow(box[0][0] - box[3][0], 2) +
                               pow(box[0][1] - box[3][1], 2)));
    
    cv::Point2f pts_dst[4];
    pts_dst[0] = cv::Point2f(0, 0);
    pts_dst[1] = cv::Point2f(crop_width, 0);
    pts_dst[2] = cv::Point2f(0, crop_height);
    pts_dst[3] = cv::Point2f(crop_width, crop_height);

    cv::Point2f pts_src[4];
    pts_src[0] = cv::Point2f(box[0][0], box[0][1]);
    pts_src[1] = cv::Point2f(box[1][0], box[1][1]);
    pts_src[2] = cv::Point2f(box[3][0], box[3][1]);
    pts_src[3] = cv::Point2f(box[2][0], box[2][1]);

    cv::Mat M = cv::getPerspectiveTransform(pts_src, pts_dst);
    cv::Mat img_crop;
    cv::warpPerspective(image, img_crop, M,
                      cv::Size(crop_width, crop_height),
                      cv::BORDER_REPLICATE);
    if (float(img_crop.rows)>float(img_crop.cols)*1.5)
        return RotateImg(img_crop);
    else
        return img_crop;
}

int CRNNRecognizer::GetBucketKey(int img_w){
    for (int bucket : this->bucket_keys_){
        if (bucket>=img_w) return bucket;
    }
    return this->bucket_keys_.back();
}

std::string CRNNRecognizer::PostProcess(NDArray pred_id, int blank){
    int pred_len = pred_id.GetShape()[1];
    std::string words="";
    for (int i=0; i<pred_len; i++){
        if (pred_id.At(0, i) != blank){
            if (i>0 && pred_id.At(0, i-1) == pred_id.At(0, i)) continue;
            else words = words + this->voc_dict_[pred_id.At(0, i)];
        }
    }
    return words;
}

// int main(){
//   std::cout<<"Hello gluon-ocr!"<<std::endl;
//   CRNNRecognizer rec("/home/sw/gluon-ocr/scripts/deploy/crnn-resnet34-symbol.json", 
//                      "/home/sw/gluon-ocr/scripts/deploy/crnn-resnet34-0000.params", 
//                      "/home/sw/demo/receipt_recognition/model/data/voc_dict_v1_7435.txt",
//                      1, 32, 1024, 1);
    
//   cv::Mat image = cv::imread("/home/sw/demo/receipt_recognition/test_imgs/20200821165522.jpg", 1);
//   std::string text = rec.Run(image);
//   std::cout<<text<<std::endl;
// }