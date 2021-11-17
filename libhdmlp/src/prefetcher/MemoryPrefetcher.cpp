#include <cstring>
#include <thread>
#include "../../include/prefetcher/MemoryPrefetcher.h"
#include "../../include/utils/MetadataStore.h"

MemoryPrefetcher::MemoryPrefetcher(std::map<std::string, std::string>& backend_options, std::vector<int>::iterator prefetch_start,
                                   std::vector<int>::iterator prefetch_end, unsigned long long int capacity, StorageBackend* backend,
                                   MetadataStore* metadata_store,
                                   int storage_level, bool alloc_buffer, Metrics* metrics) {
    if (alloc_buffer) {
        buffer = new char[capacity];
        buffer_allocated = true;
    }
    this->prefetch_start = prefetch_start;
    this->prefetch_end = prefetch_end;
    this->backend = backend;
    this->metadata_store = metadata_store;
    this->storage_level = storage_level;
    this->capacity = capacity;
    this->metrics = metrics;
    num_elems = std::distance(prefetch_start, prefetch_end);
    file_ends.resize(num_elems, 0);
    {
      int i = 0;
      for (auto iter = prefetch_start; iter != prefetch_end; ++iter, ++i) {
        int file_id = *iter;
        file_id_to_idx[file_id] = i;
        if (i == 0) {
          file_ends[i] = backend->get_file_size(file_id);
        } else {
          file_ends[i] = file_ends[i-1] + backend->get_file_size(file_id);
        }
        file_cached[file_id] = 0;
      }
    }
    this->metadata_store->store_planned_locations(prefetch_start, prefetch_end, storage_level);
}

MemoryPrefetcher::~MemoryPrefetcher() {
    if (buffer_allocated) {
        delete[] buffer;
    }
}

void MemoryPrefetcher::prefetch(int thread_id, int storage_class) {
  bool profiling = metrics != nullptr;
  std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
  while (true) {
    std::unique_lock<std::mutex> lock(prefetcher_mutex);
    int idx = prefetch_offset;
    if (idx >= num_elems) {
      // Everything is prefetched.
      break;
    }
    ++prefetch_offset;  // Claim this file to prefetch.
    const int file_id = *(prefetch_start + idx);
    if (file_cached[file_id] >= 1) {
      // A different thread cached or is caching this file.
      lock.unlock();
      continue;
    }
    file_cached[file_id] = 2;  // Mark we're prefetching it.
    lock.unlock();

    // Fetch the file without the lock.
    if (profiling) {
      t1 = std::chrono::high_resolution_clock::now();
    }
    unsigned long long int start = (idx == 0) ? 0 : file_ends[idx-1];
    backend->fetch(file_id, buffer + start);
    if (profiling) {
      t2 = std::chrono::high_resolution_clock::now();
      metrics->read_locations[storage_level][thread_id].emplace_back(OPTION_PFS);
      metrics->read_times[storage_level][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }
    // Reclaim the lock to mark the file as fetched.
    lock.lock();
    file_cached[file_id] = 1;
    lock.unlock();
    metadata_store->insert_cached_file(storage_level, file_id);
    prefetch_cv.notify_all();
  }
}

void MemoryPrefetcher::fetch_and_cache(int file_id, char* dst) {
  std::unique_lock<std::mutex> lock(prefetcher_mutex);
  if (file_cached.count(file_id) == 0) {
    throw std::runtime_error("Trying fetch_and_cache with bad file_id");
  }
  if (file_cached[file_id] == 1) {
    // A different thread already fetched this file.
    lock.unlock();
    fetch(file_id, dst);
    return;
  } else if (file_cached[file_id] == 2) {
    // A different thread is currently caching this file, wait.
    prefetch_cv.wait(lock, [&]() { return file_cached[file_id] == 1; });
    lock.unlock();
    fetch(file_id, dst);
    return;
  }
  file_cached[file_id] = 2;  // Mark we're prefetching it.
  lock.unlock();

  // Fetch without the lock.
  int idx = file_id_to_idx[file_id];
  unsigned long long int start = (idx == 0) ? 0 : file_ends[idx-1];
  unsigned long long int end = file_ends[idx];
  backend->fetch(file_id, buffer + start);
  // Copy to the destination buffer.
  memcpy(dst, buffer + start, end - start);
  // Reclaim the lock to mark the file as cached.
  lock.lock();
  file_cached[file_id] = 1;
  lock.unlock();
  metadata_store->insert_cached_file(storage_level, file_id);
  prefetch_cv.notify_all();
}

void MemoryPrefetcher::fetch(int file_id, char* dst) {
    unsigned long len;
    char* loc = get_location(file_id, &len);

    memcpy(dst, loc, len);
}

char* MemoryPrefetcher::get_location(int file_id, unsigned long* len) {
  // No lock: This is only called when the file is marked visible and cached.
  int idx = file_id_to_idx[file_id];
  unsigned long long int end = file_ends[idx];
  unsigned long long int start = (idx == 0) ? 0 : file_ends[idx-1];
  *len = end - start;
  return buffer + start;
}

int MemoryPrefetcher::get_prefetch_offset() {
    // Unsynchronized access, as this value is only used for approximating if file should be fetched remotely and
    // stale values therefore aren't critical
    return prefetch_offset;
}

bool MemoryPrefetcher::is_done() {
    return prefetch_offset >= num_elems;
}
