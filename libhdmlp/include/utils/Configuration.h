#ifndef HDMLP_CONFIGURATION_H
#define HDMLP_CONFIGURATION_H


#include <string>
#include <libconfig.h++>

class Configuration {
public:
    explicit Configuration(const std::string& config_path);

    std::string get_string_entry(const std::string& key);

    int get_int_entry(const std::string& key);

    bool get_bool_entry(const std::string& key);

    void get_storage_classes(std::vector<unsigned long long int>* capacities,
                             std::vector<int>* threads,
                             std::vector<std::map<int, int>>* bandwidths,
                             std::vector<std::string>* pf_backends,
                             std::vector<std::map<std::string, std::string>>* pf_backend_options);

    void get_pfs_bandwidth(std::map<int, int>* bandwidths);

    int get_no_distributed_threads();

    void get_bandwidths(int* networkbandwidth_clients, int* networkbandwith_filesystem);

    bool get_checkpoint();

    std::string get_checkpoint_path();

    bool get_profiling() const;

private:
    libconfig::Config cfg;

    bool profiling_enabled;

};


#endif //HDMLP_CONFIGURATION_H
