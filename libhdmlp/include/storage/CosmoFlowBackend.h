#ifndef HDMLP_COSMOFLOWBACKEND_H
#define HDMLP_COSMOFLOWBACKEND_H

#include <mutex>
#include <unordered_map>
#include "StorageBackend.h"

class CosmoFlowBackend : public StorageBackend {
public:
  CosmoFlowBackend(const std::wstring& path, int node_id,
                   bool checkpoint, std::string checkpoint_path,
                   bool cache_labels_ = true);

  void fetch_label(int file_id, char* dst) override;

  virtual int get_label_size(int file_id) override;

  int get_length() override;

  unsigned long get_file_size(int file_id) override;

  void fetch(int file_id, char* dst) override;

private:
  std::string path;
  const int sample_elem_size = 2;
  const int label_elem_size = 4;
  const int sample_num_elems = 4*128*128*128;
  const int label_num_elems = 4;
  std::vector<std::string> file_names;

  bool checkpoint;
  std::string checkpoint_path;

  bool cache_labels;
  std::unordered_map<int, std::vector<float>> label_cache;

  void init_mappings(int node_id);
  void init_mappings_from_checkpoint(int node_id);
};


#endif //HDMLP_COSMOFLOWBACKEND_H
