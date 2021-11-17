#ifndef HDMLP_FILESYSTEMPREFETCHER_H
#define HDMLP_FILESYSTEMPREFETCHER_H


#include <map>
#include <string>
#include <vector>
#include "../storage/StorageBackend.h"
#include "../utils/MetadataStore.h"
#include "MemoryPrefetcher.h"

class FileSystemPrefetcher : public MemoryPrefetcher {
public:
    FileSystemPrefetcher(std::map<std::string, std::string>& backend_options, std::vector<int>::iterator prefetch_start,
                         std::vector<int>::iterator prefetch_end, unsigned long long int capacity, StorageBackend* backend,
                         MetadataStore* metadata_store, int storage_level, int job_id, int node_id, Metrics* metrics);

    ~FileSystemPrefetcher() override;

private:
    int fd;
    std::string path;
};


#endif //HDMLP_FILESYSTEMPREFETCHER_H
