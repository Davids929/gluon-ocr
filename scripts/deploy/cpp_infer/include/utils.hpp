#pragma once

#include <iomanip>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// string split
std::vector<std::string> Split(std::string &str, const std::string delim);

// string Delect
std::string Delete(std::string &str, const std::string delim);


std::vector<std::string> ReadFile(const std::string dict_path);

std::map<std::string, std::string> LoadConfig(const std::string &config_path);