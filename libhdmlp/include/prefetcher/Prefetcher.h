#ifndef HDMLP_PREFETCHER_H
#define HDMLP_PREFETCHER_H

#include <string>
#include <deque>
#include <thread>
#include "../storage/StorageBackend.h"
#include "../utils/Sampler.h"
#include "StagingBufferPrefetcher.h"
#include "PrefetcherBackend.h"
#include "../utils/MetadataStore.h"
#include "../utils/DistributedManager.h"
#include "../utils/Metrics.h"

class Prefetcher {
public:
    Prefetcher(const std::wstring& dataset_path, const std::wstring& config_path, int batch_size, int epochs, int distr_scheme, bool drop_last_batch,
               int seed, int job_id, wchar_t** transform_names, char* transform_args, int transform_output_size, int transform_len,
               const std::wstring& filesystem_backend, const std::wstring& hdf5_data_name, const std::wstring& hdf5_target_name,
               bool collate_data);

    ~Prefetcher();

    char* get_staging_buffer();

    unsigned long long int get_next_file_end();

    void notify_data_consumed(unsigned long long int until_offset);

    int get_dataset_length();

    int get_node_id();

    int get_no_nodes();

    Metrics* metrics = nullptr;

    int largest_label_size = 0;

private:
    char* staging_buffer;
    StorageBackend* backend;
    Sampler* sampler;
    StagingBufferPrefetcher* sbf;
    PrefetcherBackend** pf_backends;
    MetadataStore* metadata_store;
    DistributedManager* distr_manager;
    TransformPipeline** transform_pipeline = nullptr;
    std::vector<int> prefetch_string;
    std::vector<std::vector<int>::iterator> storage_class_ends;
    std::vector<unsigned long long int> config_capacities;
    std::vector<int> config_no_threads;
    std::vector<std::string> config_pf_backends;
    std::vector<std::map<std::string, std::string>> config_pf_backend_options;
    std::vector<std::map<int, int>> config_bandwidths;
    std::map<int, int> config_pfs_bandwidth;
    std::vector<std::vector<std::thread>> threads;
    std::vector<std::thread> distr_threads;
    int node_id;
    int n;
    int job_id;
    int no_distributed_threads;
    int networkbandwidth_clients;
    int networkbandwidth_filesystem;
    bool checkpoint;
    std::string checkpoint_path;
    bool profiling;
    bool collate_data;

    void init_config(const std::wstring& path);

    void init_threads();

    void init_distr_threads();
};


#endif //HDMLP_PREFETCHER_H
