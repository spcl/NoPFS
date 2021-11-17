#include "../../include/storage/StorageBackendFactory.h"
#include "../../include/storage/FileSystemBackend.h"
#include "../../include/storage/HDF5Backend.h"
#include "../../include/storage/CosmoFlowBackend.h"

StorageBackend*
StorageBackendFactory::create(const std::string& storage_backend, const std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>>& dataset_path, bool checkpoint, const std::string& checkpoint_path,
                              int node_id,
                              const std::string& hdf5_data_name,
                              const std::string& hdf5_target_name) {
    if (storage_backend == "filesystem") {
        return new FileSystemBackend(dataset_path, checkpoint, checkpoint_path, node_id);
    } else if (storage_backend == "hdf5") {
      return new HDF5Backend(dataset_path, node_id,
                             checkpoint, checkpoint_path,
                             hdf5_data_name, hdf5_target_name);
    } else if (storage_backend == "cosmoflow") {
      return new CosmoFlowBackend(dataset_path, node_id,
                                  checkpoint, checkpoint_path);
    } else {
        throw std::runtime_error("Unsupported storage backend");
    }
}
