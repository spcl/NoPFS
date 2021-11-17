#ifndef HDMLP_DISTRIBUTEDMANAGER_H
#define HDMLP_DISTRIBUTEDMANAGER_H

// How many elements do we need to be ahead to decide that a file is available at a remote location, see get_remote_storage_class for details
#define REMOTE_PREFETCH_OFFSET_DIFF 20

// Tag that is used to indicate MPI requests, shall not be used in responses to requests
#define REQUEST_TAG 0

#include "MetadataStore.h"
#include "../prefetcher/PrefetcherBackend.h"
#include "../storage/StorageBackend.h"
#include <vector>
#include <atomic>
#include <mpi.h>

class DistributedManager {
public:
    DistributedManager(MetadataStore* metadata_store, PrefetcherBackend** prefetcher_backends);

    int get_no_nodes() const;

    int get_node_id() const;

    void serve();

    bool fetch(int file_id, char* dst, int thread_id);

    int get_remote_storage_class(int file_id);

    void distribute_prefetch_strings(std::vector<int>* local_prefetch_string,
                                     std::vector<std::vector<int>::iterator>* storage_class_ends,
                                     int num_storage_classes);

    int generate_and_broadcast_seed();

    void stop_all_threads(int num_threads);

    void set_backend(StorageBackend* pBackend);

private:
    struct FileAvailability {
        int node_id;
        int storage_class;
        int offset;
    };
    MPI_Comm JOB_COMM;
    PrefetcherBackend** pf_backends = nullptr;
    MetadataStore* metadata_store;
    StorageBackend* storage_backend;
    int n;
    int node_id;
    bool has_initialized_mpi = false;
    std::unordered_map<int, FileAvailability> file_availability; // Stores mapping of file_id -> availability info
    std::atomic<bool> stop_flag;

    void parse_received_prefetch_data(int* rcv_data, int arr_size, int global_max_size);
};


#endif //HDMLP_DISTRIBUTEDMANAGER_H
