#ifndef HDMLP_METADATASTORE_H
#define HDMLP_METADATASTORE_H

#define OPTION_PFS 0
#define OPTION_REMOTE 1
#define OPTION_LOCAL 2

#include <unordered_map>
#include <shared_mutex>
#include <map>
#include <vector>

class MetadataStore {
public:
    MetadataStore(int networkbandwidth_clients, int networkbandwidth_filesystem,
                  std::map<int, int>* pfs_bandwidths, std::vector<std::map<int, int>>* storage_level_bandwidths,
                  std::vector<int>* no_threads);

    void insert_cached_file(int storage_level, int file_id);

    int get_storage_level(int file_id);

    void set_no_nodes(int n);

    void get_option_order(int local_storage_level, int remote_storage_level, int* options);

  void store_planned_locations(std::vector<int>::iterator& start,
                               std::vector<int>::iterator& end,
                               int storage_level);

  int get_planned_storage_level(int file_id);

private:
  std::unordered_map<int, int> planned_file_locations;
    std::unordered_map<int, int> file_locations;
    std::shared_timed_mutex file_locations_mutex;
  std::shared_timed_mutex planned_file_locations_mutex;
    std::vector<double> interp_storage_level_bandwidths;
    double interp_pfs_bandwidth;
    int n;
    double networkbandwidth_clients;
    double networkbandwidth_filesystem;

    static double interpolate_map(int key_val, std::map<int, int>* map);
};


#endif //HDMLP_METADATASTORE_H
