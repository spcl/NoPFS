#ifndef HDMLP_MEMORYPREFETCHER_H
#define HDMLP_MEMORYPREFETCHER_H


#include <map>
#include <string>
#include <vector>
#include <condition_variable>
#include "PrefetcherBackend.h"
#include "../storage/StorageBackend.h"
#include "../utils/MetadataStore.h"
#include "../utils/Metrics.h"

class MemoryPrefetcher : public PrefetcherBackend {
public:
    MemoryPrefetcher(std::map<std::string, std::string>& backend_options, std::vector<int>::iterator prefetch_start,
                     std::vector<int>::iterator prefetch_end, unsigned long long int capacity, StorageBackend* backend, MetadataStore* metadata_store,
                     int storage_level, bool alloc_buffer, Metrics* metrics);

    ~MemoryPrefetcher() override;

    void prefetch(int thread_id, int storage_class) override;

    void fetch(int file_id, char* dst) override;

    void fetch_and_cache(int file_id, char* dst) override;

    char* get_location(int file_id, unsigned long* len) override;

    int get_prefetch_offset() override;

    bool is_done() override;

protected:
    char* buffer;
    std::vector<unsigned long long int> file_ends;
    std::unordered_map<int, int> file_id_to_idx;
    // 0 = not cached; 1 == cached; 2 == caching
    std::unordered_map<int, char> file_cached;
    StorageBackend* backend;
    MetadataStore* metadata_store;
    Metrics* metrics;
    std::vector<int>::iterator prefetch_start;
    std::vector<int>::iterator prefetch_end;
    std::mutex prefetcher_mutex;
    std::condition_variable prefetch_cv;
    int num_elems;
    int prefetch_offset = 0;
    int storage_level;
    unsigned long long capacity;
    bool buffer_allocated = false;
};


#endif //HDMLP_MEMORYPREFETCHER_H
