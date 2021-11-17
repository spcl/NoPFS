#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <iostream>
#include "../../include/prefetcher/FileSystemPrefetcher.h"
#include "../../include/utils/MetadataStore.h"

FileSystemPrefetcher::FileSystemPrefetcher(std::map<std::string, std::string>& backend_options, std::vector<int>::iterator prefetch_start,
                                           std::vector<int>::iterator prefetch_end, unsigned long long int capacity, StorageBackend* backend,
                                           MetadataStore* metadata_store, int storage_level, int job_id, int node_id, Metrics* metrics) :
        MemoryPrefetcher(backend_options, prefetch_start, prefetch_end, capacity, backend, metadata_store, storage_level, false, metrics) {
    path = backend_options["path"];
    if (path == "env") {
      char* env_path = std::getenv("HDMLP_FILESYSTEM_PATH");
      if (env_path == nullptr) {
        throw std::runtime_error("HDMLP_FILESYSTEM_PATH not set");
      }
      path = env_path;
    }
    if (path.back() != '/') {
        path += '/';
    }
    struct stat base_dir{};
    if (!(stat(path.c_str(), &base_dir) == 0 && S_ISDIR(base_dir.st_mode))) { // NOLINT(hicpp-signed-bitwise)
        throw std::runtime_error("Configured file system prefetching path doesn't exist or isn't a directory");
    }

    path = path + std::to_string(job_id) + "_" + std::to_string(node_id);

    if ((fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600)) < 0) { // NOLINT(hicpp-signed-bitwise)
        throw std::runtime_error("Error opening file for prefetching");
    }
    if (lseek(fd, capacity - 1, SEEK_SET) < 0) {
        throw std::runtime_error("Error seeking file for prefetching");
    }
    if (write(fd, "", 1) != 1) {
        throw std::runtime_error("Error writing to file for prefetching");
    }
    buffer = static_cast<char*>(mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)); // NOLINT(hicpp-signed-bitwise)
    if (buffer == MAP_FAILED) {
        throw std::runtime_error("Error while mmapping file");
    }
}

FileSystemPrefetcher::~FileSystemPrefetcher() {
    munmap(buffer, capacity);
    close(fd);
    unlink(path.c_str());
}
