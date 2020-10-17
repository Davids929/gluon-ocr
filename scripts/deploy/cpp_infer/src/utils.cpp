
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
        std::cout << "no such label file: " << dict_path << std::endl;
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
