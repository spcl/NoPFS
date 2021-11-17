#include <iostream>
#include "../../include/prefetcher/PrefetcherBackendFactory.h"
#include "../../include/prefetcher/MemoryPrefetcher.h"
#include "../../include/prefetcher/FileSystemPrefetcher.h"
#include "../../include/utils/Metrics.h"

PrefetcherBackend* PrefetcherBackendFactory::create(const std::string& prefetcher_backend, std::map<std::string, std::string>& backend_options,
                                                    unsigned long long int capacity,
                                                    std::vector<int>::iterator start, std::vector<int>::iterator end, StorageBackend* storage_backend,
                                                    MetadataStore* metadata_store,
                                                    int storage_level, int job_id, int node_id, Metrics* metrics) {
    PrefetcherBackend* pfb;
    if (prefetcher_backend == "memory") {
        pfb = new MemoryPrefetcher(backend_options, start, end, capacity, storage_backend, metadata_store,
                                   storage_level, true, metrics);
    } else if (prefetcher_backend == "filesystem") {
        pfb = new FileSystemPrefetcher(backend_options, start, end, capacity, storage_backend, metadata_store,
                                       storage_level, job_id, node_id, metrics);
    } else {
        throw std::runtime_error("Unsupported prefetch backend");
    }
    return pfb;
}
