#ifndef HDMLP_STORAGEBACKENDFACTORY_H
#define HDMLP_STORAGEBACKENDFACTORY_H


#include "StorageBackend.h"

class StorageBackendFactory {
public:
  static StorageBackend* create(const std::string& storage_backend, const std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>>& dataset_path, bool checkpoint, const std::string& checkpoint_path, int node_id,
                                const std::string& hdf5_data_name,
                                const std::string& hdf5_target_name);
};


#endif //HDMLP_STORAGEBACKENDFACTORY_H
