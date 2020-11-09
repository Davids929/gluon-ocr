
#include "include/config.hpp"
#include "include/ocr_det.hpp"
#include "include/ocr_rec.hpp"

void PrintUsage(){
    std::cout << "Usage:" << std::endl;
    std::cout << " <config file path>"<<std::endl
              << "[--only_recog <Specify this option if recognize image patch>]"<<std::endl
              << "--gpu_id <runing gpu id>" <<std::endl
              << "--det_symbol <detect model symbol file in json format>  " << std::endl
              << "--det_params <detect model params file> " << std::endl
              << "--db_thresh  <threshold of db detect model> " << std::endl
              << "--db_box_threshold <box threshold of db detect model> " << std::endl
              << "--db_unclip_ratio < unclip_ratio of db detect model> " << std::endl
              << "--det_max_side <max input side of detect model> " << std::endl
              << "--det_min_side <min input side of detect model> " << std::endl

              << "--rec_symbol <recog model symbol file in json format>  " << std::endl
              << "--rec_params <recog model params file> " << std::endl
              << "--dict_path <vocabulary file> " << std::endl
              << "--rec_max_side <max input side of recog model> " << std::endl
              << "--rec_short_side <short side length of recog model> " << std::endl
              << "--num_buckets <bucket number> " << std::endl
              << "--image <path to the image used for prediction> " << std::endl
              
              << "[--visualize  <Specify this option if show ocr results>]"
              << std::endl;
}

void RunDemo(std::string img_path, Config config){
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);

    DBDetector ocr_det(config.det_model_path, 
                       config.det_params_path,
                       config.gpu_id,
                       config.db_max_side_len,
                       config.db_min_side_len,
                       config.db_thresh,
                       config.db_box_thresh,
                       config.db_unclip_ratio,
                       config.visualize);

    CRNNRecognizer ocr_rec(config.rec_model_path,
                           config.rec_params_path,
                           config.voc_dict_path,
                           config.gpu_id,
                           config.rec_max_side_len,
                           config.rec_short_side,
                           config.num_buckets);

    auto start = std::chrono::system_clock::now();
    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector<std::string> texts;
    if (config.only_recog){
        std::string text = ocr_rec.Run(image);
    }    
    else{
        ocr_det.Run(image, boxes);
        ocr_rec.Run(image, boxes, texts);
    }
    auto end = std::chrono::system_clock::now();
    std::cout<<"cost time:"<<std::chrono::duration<double, std::milli>(end - start).count()/1000<<" s"<<std::endl;
    if (config.visualize){
        viz::PlotRect(image, boxes);
        cv::imwrite("./results.jpg", image);
    }
}

int main(int argc, char **argv){
    
    int index = 1;
    std::string image_path = "";
    if (argc < 3) {
        std::cerr << "[ERROR] usage: " << argv[0]
                << " config_filepath --image imagepath\n";
        exit(1);
    }
    Config config(argv[1]);
    index++;
    
    while (index < argc) {
        if (strcmp("--only_recog", argv[index]) == 0) {
            config.only_recog = true;
        }else if (strcmp("--gpu_id", argv[index]) == 0) {
            index++;
            config.gpu_id = std::stoi(index < argc ? argv[index]:"-1");
        //detect model config
        } else if (strcmp("--det_symbol", argv[index]) == 0) {
            index++;
            config.det_model_path = (index < argc ? argv[index]:"");
        } else if (strcmp("--det_params", argv[index]) == 0) {
            index++;
            config.det_params_path = (index < argc ? argv[index]:"");
        } else if (strcmp("--db_thresh", argv[index]) == 0) {
            index++;
            config.db_thresh = std::stoi(index < argc ? argv[index]:"0.3");
        } else if (strcmp("--db_box_thresh", argv[index]) == 0) {
            index++;
            config.db_box_thresh = std::stoi(index < argc ? argv[index]:"0.5");
        } else if (strcmp("--db_unclip_ratio", argv[index]) == 0) {
            index++;
            config.db_unclip_ratio = std::stoi(index < argc ? argv[index]:"1.6");
        } else if (strcmp("--det_max_side", argv[index]) == 0) {
            index++;
            config.db_max_side_len = std::stoi(index < argc ? argv[index]:"1024");
        }else if (strcmp("--det_min_side", argv[index]) == 0) {
            index++;
            config.db_min_side_len = std::stoi(index < argc ? argv[index]:"640");

        //recog model config
        } else if (strcmp("--rec_symbol", argv[index]) == 0) {
            index++;
            config.rec_model_path = (index < argc ? argv[index]:"");
        } else if (strcmp("--rec_params", argv[index]) == 0) {
            index++;
            config.rec_params_path = (index < argc ? argv[index]:"");
        } else if (strcmp("--dict_path", argv[index]) == 0) {
            index++;
            config.voc_dict_path = (index < argc ? argv[index]:"");
        } else if (strcmp("--rec_max_side", argv[index]) == 0) {
            index++;
            config.rec_max_side_len = std::stoi(index < argc ? argv[index]:"1024");
        } else if (strcmp("--rec_short_side", argv[index]) == 0) {
            index++;
            config.rec_short_side = std::stoi(index < argc ? argv[index]:"32");
        } else if (strcmp("--num_buckets", argv[index]) == 0) {
            index++;
            config.num_buckets = std::stoi(index < argc ? argv[index]:"1");
        // 
        } else if (strcmp("--image", argv[index]) == 0) {
            index++;
            image_path = (index < argc ? argv[index]:"");
        } else if (strcmp("--visualize", argv[index]) == 0) {
            config.visualize = true;
        } else if (strcmp("--help", argv[index]) == 0) {
            PrintUsage();
            return 0;
        }
        index++;
    }
    
    if (image_path.empty()) {
        std::cout << "ERROR: Path to the input image is not specified."<<std::endl;
        return 1;
    }

    if (config.det_model_path.empty() || config.det_params_path.empty() || 
        config.rec_model_path.empty() || config.rec_params_path.empty() ||
        config.voc_dict_path.empty()) {
        std::cout << "ERROR: Model details such as symbol, param and/or dict files are not specified"<<std::endl;
        return 1;
    }
    
    RunDemo(image_path, config);
return 0;
}