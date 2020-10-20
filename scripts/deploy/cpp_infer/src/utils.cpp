
#include "include/utils.hpp"

std::vector<std::string> Split(std::string &str, const std::string delim){
  std::vector<std::string> res;
  std::size_t pos, last_pos;

  last_pos = str.find_first_not_of(delim, 0);
  pos      = str.find(delim, last_pos);
  while (last_pos != std::string::npos){
    res.push_back(str.substr(last_pos, pos-last_pos));
    last_pos = str.find_first_not_of(delim, pos);
    pos = str.find(delim, last_pos);
  }
  return res;
}

std::string Delete(std::string &str, const std::string delim){
  std::size_t pos = 0;
  pos = str.find_first_of(delim, pos);
  if (pos == std::string::npos)
    return str;
  return Delete(str.erase(pos, 1), delim);
}

std::vector<std::string> ReadFile(const std::string dict_path){
    std::ifstream in(dict_path);
    std::string line;
    std::vector<std::string> m_vec;
    if (in) {
        while (getline(in, line)) {
            m_vec.push_back(line);
        }
    } else {
        std::cout << "no such file: " << dict_path << std::endl;
        exit(1);
    }
    return m_vec;
}

std::map<std::string, std::string> LoadConfig(const std::string &config_path){
  
  std::vector<std::string> lines = ReadFile(config_path); 
  std::map<std::string, std::string> dict;
  for (int i = 0; i < lines.size(); i++) {
    // pass for empty line or comment
    if (lines[i].size() <= 1 || lines[i][0] == '#') {
      continue;
    }
    std::string line = Delete(lines[i], " ");
    std::vector<std::string> res = Split(line, "=");
    if (res.size()>=2)
      dict[res[0]] = res[1];
  }
  return dict;
}



//Transformation of two-dimensional matrix into vector.
std::vector<std::vector<float>> Mat2Vector(cv::Mat mat){
    std::vector<std::vector<float>> img_vec;
    std::vector<float> tmp;

    for (int i = 0; i < mat.rows; ++i) {
        tmp.clear();
        for (int j = 0; j < mat.cols; ++j) {
        tmp.push_back(mat.at<float>(i, j));
        }
        img_vec.push_back(tmp);
    }
    return img_vec;
}


//Sorts the four points of the rectangle clockwise
std::vector<std::vector<int>> 
OrderPointsClockwise(std::vector<std::vector<int>> pts) {
  std::vector<std::vector<int>> box = pts;
  std::sort(box.begin(), box.end(), Xsort<int>);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1])
    std::swap(leftmost[0], leftmost[1]);

  if (rightmost[0][1] > rightmost[1][1])
    std::swap(rightmost[0], rightmost[1]);

  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  return rect;
}