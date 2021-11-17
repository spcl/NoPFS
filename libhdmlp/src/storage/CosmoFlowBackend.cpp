#include <dirent.h>
#include <codecvt>
#include <locale>
#include <H5Cpp.h>
#include <H5File.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "../../include/storage/CosmoFlowBackend.h"
#include <cstring>      // strrchr, strcmp
#include <algorithm>    // std::sort
#include <cstring>
#include <stdio.h>


CosmoFlowBackend::CosmoFlowBackend(const std::wstring& path_, int node_id,
                                   bool checkpoint_, std::string checkpoint_path_,
                                   bool cache_labels_) :
  checkpoint(checkpoint_), cache_labels(cache_labels_) {
  using type = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<type, wchar_t> converter;
  std::string str_path = converter.to_bytes(path_);
  if (str_path.back() != '/') {
    str_path += '/';
  }
  path = str_path;
  if (checkpoint) {
    if (checkpoint_path_ == "") {
      checkpoint_path = this->path;
    } else {
      checkpoint_path = checkpoint_path_;
    }
    init_mappings_from_checkpoint(node_id);
  } else {
    init_mappings(node_id);
  }
}

int CosmoFlowBackend::get_length() {
    return file_names.size();
}

unsigned long CosmoFlowBackend::get_file_size(int file_id) {
  return sample_elem_size * sample_num_elems;
}

void CosmoFlowBackend::fetch(int file_id, char* dst) {
  FILE* f = fopen((path + file_names[file_id]).c_str(), "rb");
  fseek(f, label_elem_size*label_num_elems, SEEK_SET);  // Skip label.
  fread(dst, sample_elem_size, sample_num_elems, f);
  fclose(f);
}

void CosmoFlowBackend::init_mappings(int node_id) {
  struct dirent* entry = nullptr;
  DIR* dp = nullptr;

  dp = opendir(path.c_str());
  if (dp == nullptr) {
    throw std::runtime_error("Invalid path specified");
  }
  if (node_id == 0) {
    std::cout << "Scanning binary files" << std::endl;
  }
  int file_id = 0;
  while ((entry = readdir(dp))) {
    char* ext = strrchr(entry->d_name, '.');
    if (ext && strcmp(ext, ".bin") == 0) {
      std::string file_name = entry->d_name;
      file_names.push_back(file_name);
      if (cache_labels) {
        std::vector<float> label_data(4);
        FILE* f = fopen((path + file_name).c_str(), "rb");
        fread(label_data.data(), label_elem_size, label_num_elems, f);
        fclose(f);
        label_cache.emplace(file_id, label_data);
      }
      ++file_id;
      if (node_id == 0 && (file_id % 10000 == 0)) {
        std::cout << "Scanned " << file_id << " files" << std::endl;
      }
    }
  }
  std::ofstream checkpoint_stream;
  if (checkpoint && node_id == 0) {
    std::cout << "Saving binary metadata" << std::endl;
    checkpoint_stream.open(checkpoint_path + "/hdmlp_metadata_",
                           std::ofstream::out | std::ofstream::trunc);
    for (size_t i = 0; i < file_names.size(); ++i) {
      checkpoint_stream << file_names[i];
      if (cache_labels) {
        for (const auto& v : label_cache[i]) {
          checkpoint_stream << "," << v;
        }
      }
      checkpoint_stream << "\n";
    }
    checkpoint_stream.close();
    std::rename((checkpoint_path + "/hdmlp_metadata_").c_str(),
                (checkpoint_path + "/hdmlp_metadata").c_str());
  }
}

void CosmoFlowBackend::init_mappings_from_checkpoint(int node_id) {
  std::ifstream checkpoint_stream(checkpoint_path + "/hdmlp_metadata");
  if (checkpoint_stream.fail()) {
    this->init_mappings(node_id);
  }
  std::string line;
  int file_id = 0;
  while (std::getline(checkpoint_stream, line)) {
    std::stringstream ss(line);
    if (cache_labels) {
      std::string file_name;
      std::getline(ss, file_name, ',');
      file_names.push_back(file_name);
      ss.ignore();  // Drop comma.
      std::vector<float> label_data(label_num_elems);
      for (int i = 0; i < label_num_elems; ++i) {
        ss >> label_data[i];
        ss.ignore();  // Drop comma.
      }
      label_cache.emplace(file_id, label_data);
      ++file_id;
    } else {
      file_names.push_back(line);  // Line is just the filename.
    }
  }
}

int CosmoFlowBackend::get_label_size(int file_id) {
  return label_elem_size * label_num_elems;
}

void CosmoFlowBackend::fetch_label(int file_id, char* dst) {
  if (cache_labels) {
    std::memcpy(dst, label_cache[file_id].data(),
                label_elem_size * label_num_elems);
  } else {
    FILE* f = fopen((path + file_names[file_id]).c_str(), "rb");
    fread(dst, label_elem_size, label_num_elems, f);
    fclose(f);
  }
}
