#include "../../include/utils/MetadataStore.h"
#include <algorithm>
#include <climits>


MetadataStore::MetadataStore(int networkbandwidth_clients, // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
                             int networkbandwidth_filesystem,
                             std::map<int, int>* pfs_bandwidths,
                             std::vector<std::map<int, int>>* storage_level_bandwidths,
                             std::vector<int>* no_threads) {
    interp_pfs_bandwidth = interpolate_map(n, pfs_bandwidths);
    for (int unsigned long i = 0; i < no_threads->size(); i++) {
        int threads = (*no_threads)[i];
        std::map<int, int>* storage_level_bandwidth = &(*storage_level_bandwidths)[i];
        interp_storage_level_bandwidths.push_back(interpolate_map(threads, storage_level_bandwidth) / threads);
    }
    this->networkbandwidth_clients = networkbandwidth_clients;
    this->networkbandwidth_filesystem = networkbandwidth_filesystem;
}

void MetadataStore::set_no_nodes(int no_nodes) {
    this->n = no_nodes;
}

void MetadataStore::insert_cached_file(int storage_level, int file_id) {
    std::lock_guard<std::shared_timed_mutex> writer_lock(file_locations_mutex);
    file_locations[file_id] = storage_level;
}

int MetadataStore::get_storage_level(int file_id) {
    std::shared_lock<std::shared_timed_mutex> reader_lock(file_locations_mutex);
    if (file_locations.count(file_id) != 0) {
        return file_locations[file_id];
    } else {
        return 0;
    }
}

void MetadataStore::store_planned_locations(std::vector<int>::iterator& start,
                                            std::vector<int>::iterator& end,
                                            int storage_level) {
  std::lock_guard<std::shared_timed_mutex> lock(planned_file_locations_mutex);
  for (auto i = start; i < end; ++i) {
    planned_file_locations[*i] = storage_level;
  }
}

int MetadataStore::get_planned_storage_level(int file_id) {
  std::shared_lock<std::shared_timed_mutex> lock(planned_file_locations_mutex);
  if (planned_file_locations.count(file_id) != 0) {
    return planned_file_locations[file_id];
  } else {
    return -1;
  }
}

double MetadataStore::interpolate_map(int key_val, std::map<int, int>* map) {
    if (map->count(key_val) > 0) {
        return (*map)[key_val];
    } else {
        int lb = 0;
        int ub = INT_MAX;
        int min_nodes = INT_MAX;
        int max_nodes = 0;
        for (auto& pair : *map) {
            if (pair.first > lb && pair.first < key_val) {
                lb = pair.first;
            } else if (pair.first < ub && pair.first > key_val) {
                ub = pair.first;
            }
            if (pair.first < min_nodes) {
                min_nodes = pair.first;
            }
            if (pair.first > max_nodes) {
                max_nodes = pair.first;
            }
        }
        if (lb == 0) {
            return (*map)[min_nodes];
        } else if (ub == INT_MAX) {
            return (*map)[max_nodes];
        } else {
            // Interpolate
            int lb_bandwidth = (*map)[lb];
            int ub_bandwidth = (*map)[ub];
            return lb_bandwidth + (double) (key_val - lb) * (ub_bandwidth - lb_bandwidth) / (ub - lb);
        }
    }
}

/**
 * Returns prefetching options, ordered according to the bandwidth.
 * Option 0 is PFS, 1 remote read and 2 local read. See the performance model for details
 * @param local_storage_level In which local storage level the file is available, 0 if not
 * @param remote_storage_level In which remote storage level the file is available, 0 if not
 * @param options Array with 3 elements, options will be put into this array
 */
void MetadataStore::get_option_order(int local_storage_level, int remote_storage_level, int* options) {
    double pfs_speed = std::min(interp_pfs_bandwidth / n, networkbandwidth_filesystem);
    double local_speed = 0.0;
    double remote_speed = 0.0;
    if (local_storage_level != 0) {
        local_speed = interp_storage_level_bandwidths[local_storage_level];
    }
    if (remote_storage_level != 0) {
        remote_speed = std::min(interp_storage_level_bandwidths[remote_storage_level], networkbandwidth_clients);
    }
    double speeds[] = {pfs_speed, remote_speed, local_speed};
    options[0] = 0;
    options[1] = 1;
    options[2] = 2;
    std::sort(options, options + 3, [&speeds](int& a, int& b) {
        if (speeds[a] == speeds[b]) {
            // Prefer local over remote over PFS in case of ties
            return (a == OPTION_LOCAL) || (a == OPTION_REMOTE && b != OPTION_LOCAL);
        }
        return speeds[a] > speeds[b];
    });
}
