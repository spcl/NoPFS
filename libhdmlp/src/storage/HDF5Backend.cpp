#include <dirent.h>
#include <codecvt>
#include <locale>
#include <H5Cpp.h>
#include <H5File.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "../../include/storage/HDF5Backend.h"
#include <cstring>      // strrchr, strcmp
#include <algorithm>    // std::sort
#include <cstring>


std::mutex HDF5Backend::hdf5_lock;


HDF5Backend::HDF5Backend(const std::wstring& path, int node_id,
                         bool checkpoint, std::string checkpoint_path,
                         const std::string& data_name,
                         const std::string target_name,
                         bool cache_labels_) :
  dataset_name_sample(data_name), dataset_name_label(target_name),
  cache_labels(cache_labels_) {
    using type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<type, wchar_t> converter;
    std::string str_path = converter.to_bytes(path);
    if (str_path.back() != '/') {
        str_path += '/';
    }
    this->path = str_path;
    this->checkpoint = checkpoint;
    if (checkpoint) {
      if (checkpoint_path == "") {
        this->checkpoint_path = this->path;
      } else {
        this->checkpoint_path = checkpoint_path;
      }
      this->init_mappings_from_checkpoint(node_id);
    } else {
      init_mappings(node_id);
    }
}

int HDF5Backend::get_length() {
    return file_names.size();
}

unsigned long HDF5Backend::get_file_size(int file_id) {
    return sample_num_elems[file_id] * sample_elem_size;
}

void HDF5Backend::fetch(int file_id, char* dst) {
  std::lock_guard<std::mutex> lg(hdf5_lock);
  //std::cout << "HDF5 fetching " << file_id << std::endl;
    H5::H5File file(path + file_names[file_id], H5F_ACC_RDONLY);
    H5::DataSet sample = file.openDataSet(dataset_name_sample);
    H5::DataSpace sample_dataspace = sample.getSpace();
    int sample_rank = sample_dataspace.getSimpleExtentNdims();
    hsize_t sample_dims[sample_rank];
    sample_dataspace.getSimpleExtentDims(sample_dims, NULL);
    H5::DataSpace memspace(sample_rank, sample_dims);
    sample.read(dst, H5::PredType::NATIVE_FLOAT, memspace, sample_dataspace);
    //std::cout << "HDF5 done fetching " << file_id << std::endl;
}

void HDF5Backend::init_mappings(int node_id) {
    std::vector<FileInformation> file_metadata;
    struct dirent* entry = nullptr;
    DIR* dp = nullptr;

    dp = opendir(this->path.c_str());
    if (dp == nullptr) {
        throw std::runtime_error("Invalid path specified");
    }
    if (node_id == 0) {
      std::cout << "Scanning HDF5 files" << std::endl;
    }
    int file_id = 0;
    while ((entry = readdir(dp))) {
        char* ext = strrchr(entry->d_name, '.');
        if (ext && (strcmp(ext, ".hdf5") == 0 || strcmp(ext, ".h5") == 0)) {
          std::lock_guard<std::mutex> lg(hdf5_lock);
            FileInformation fi;
            fi.file_name = entry->d_name;
            H5::H5File file(this->path + entry->d_name, H5F_ACC_RDONLY);
            H5::DataSet sample = file.openDataSet(dataset_name_sample);
            H5::DataSet label = file.openDataSet(dataset_name_label);
            H5T_class_t sample_type_class = sample.getTypeClass();
            H5T_class_t label_type_class = label.getTypeClass();
            if (sample_type_class == H5T_INTEGER) {
                sample_elem_size = sizeof(int);
            } else if (sample_type_class == H5T_FLOAT) {
                sample_elem_size = sizeof(float);
            } else {
                throw std::runtime_error("Unsupported HDF5 datatype");
            }
            if (label_type_class == H5T_INTEGER) {
                label_elem_size = sizeof(int);
            } else if (label_type_class == H5T_FLOAT) {
                label_elem_size = sizeof(float);
            } else {
                throw std::runtime_error("Unsupported HDF5 datatype");
            }
            H5::DataSpace sample_dataspace = sample.getSpace();
            H5::DataSpace label_dataspace = label.getSpace();
            int sample_rank = sample_dataspace.getSimpleExtentNdims();
            int label_rank = label_dataspace.getSimpleExtentNdims();
            hsize_t sample_dims[sample_rank];
            hsize_t label_dims[label_rank];
            sample_dataspace.getSimpleExtentDims(sample_dims, NULL);
            label_dataspace.getSimpleExtentDims(label_dims, NULL);
            int sample_size = 1;
            for (int i = 0; i < sample_rank; i++) {
                sample_size *= sample_dims[i];
            }
            int label_size = 1;
            for (int i = 0; i < label_rank; i++) {
                label_size *= label_dims[i];
            }
            fi.num_elems_sample = sample_size;
            fi.num_elems_label = label_size;
            file_metadata.emplace_back(fi);
            if (cache_labels) {
              std::vector<float> label_data(4);
              H5::DataSpace memspace(label_rank, label_dims);
              label.read(label_data.data(), H5::PredType::NATIVE_FLOAT,
                         memspace, label_dataspace);
              label_cache.emplace(file_id, label_data);
            }
            ++file_id;
        }
    }
    std::sort(file_metadata.begin(), file_metadata.end(), [](FileInformation& a, FileInformation& b) {
                  return a.file_name > b.file_name;
              }
    );
    std::ofstream checkpoint_stream;
    if (checkpoint && node_id == 0) {
      std::cout << "Saving HDF5 metadata" << std::endl;
      checkpoint_stream.open(checkpoint_path + "/hdmlp_metadata_",
                             std::ofstream::out | std::ofstream::trunc);
      // Save sample and label element sizes.
      checkpoint_stream << sample_elem_size << "," << label_elem_size << "\n";
    }
    file_id = 0;
    for (auto & i : file_metadata) {
        file_names.emplace_back(i.file_name);
        sample_num_elems.emplace_back(i.num_elems_sample);
        label_num_elems.emplace_back(i.num_elems_label);
        if (checkpoint && node_id == 0) {
          checkpoint_stream << i.file_name << ","
                            << i.num_elems_sample << ","
                            << i.num_elems_label;
          if (cache_labels) {
            for (const auto& v : label_cache[file_id]) {
              checkpoint_stream << "," << v;
            }
          }
          checkpoint_stream << "\n";
          ++file_id;
        }
    }
    if (checkpoint && node_id == 0) {
      checkpoint_stream.close();
      std::rename((checkpoint_path + "/hdmlp_metadata_").c_str(),
                  (checkpoint_path + "/hdmlp_metadata").c_str());
    }
}

void HDF5Backend::init_mappings_from_checkpoint(int node_id) {
  std::ifstream checkpoint_stream(checkpoint_path + "/hdmlp_metadata");
  if (checkpoint_stream.fail()) {
    this->init_mappings(node_id);
  }
  std::string line;
  // Load the sample and label element sizes.
  {
    std::getline(checkpoint_stream, line);
    std::stringstream ss(line);
    ss >> sample_elem_size;
    ss.ignore();  // Drop comma.
    ss >> label_elem_size;
  }
  int file_id = 0;
  while (std::getline(checkpoint_stream, line)) {
    std::stringstream ss(line);
    std::string file_name;
    int num_elems_sample, num_elems_label;
    std::getline(ss, file_name, ',');
    ss >> num_elems_sample;
    ss.ignore();  // Drop comma.
    ss >> num_elems_label;
    file_names.push_back(file_name);
    sample_num_elems.push_back(num_elems_sample);
    label_num_elems.push_back(num_elems_label);
    if (cache_labels) {
      std::vector<float> label_data(num_elems_label);
      for (int i = 0; i < num_elems_label; ++i) {
        ss >> label_data[i];
        ss.ignore();  // Drop comma.
      }
      label_cache.emplace(file_id, label_data);
    }
    ++file_id;
  }
}

int HDF5Backend::get_label_size(int file_id) {
    return label_num_elems[file_id] * label_elem_size;
}

void HDF5Backend::fetch_label(int file_id, char* dst) {
  if (cache_labels) {
    std::memcpy(dst, label_cache[file_id].data(),
                label_num_elems[file_id]*label_elem_size);
  } else {
    std::lock_guard<std::mutex> lg(hdf5_lock);
    //std::cout << "HDF5 fetching label " << file_id << std::endl;
    H5::H5File file(path + file_names[file_id], H5F_ACC_RDONLY);
    H5::DataSet label = file.openDataSet(dataset_name_label);
    H5::DataSpace label_dataspace = label.getSpace();
    int label_rank = label_dataspace.getSimpleExtentNdims();
    hsize_t label_dims[label_rank];
    label_dataspace.getSimpleExtentDims(label_dims, NULL);
    H5::DataSpace memspace(label_rank, label_dims);
    label.read(dst, H5::PredType::NATIVE_FLOAT, memspace, label_dataspace);
    //std::cout << "HDF5 done fetching label " << file_id << std::endl;
  }
}
