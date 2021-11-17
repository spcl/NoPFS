#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <cstdlib>
#include <string.h>
#include "../../include/utils/Configuration.h"

Configuration::Configuration(const std::string& config_path) {
    try {
        cfg.readFile(config_path.c_str());
    } catch (const libconfig::FileIOException& fioex) {
        throw std::runtime_error("I/O error while reading config file.");
    } catch (const libconfig::ParseException& pex) {
        std::ostringstream error;
        error << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
        throw std::runtime_error(error.str());
    }
    const char* hdmlp_profiling = std::getenv("HDMLPPROFILING");
    profiling_enabled = !(!hdmlp_profiling || strcmp(hdmlp_profiling, "0") == 0 || strcmp(hdmlp_profiling, "false") == 0);
}

std::string Configuration::get_string_entry(const std::string& key) {
    std::string val;
    try {
        val = cfg.lookup(key).c_str();
    } catch (const libconfig::SettingNotFoundException& nfex) {
        val = "";
    }
    return val;
}

int Configuration::get_int_entry(const std::string& key) {
    int val;
    try {
        val = cfg.lookup(key);
    } catch (const libconfig::SettingNotFoundException& nfex) {
        val = -1;
    }
    return val;
}

bool Configuration::get_bool_entry(const std::string& key) {
    bool val;
    try {
        val = cfg.lookup(key);
    } catch (const libconfig::SettingNotFoundException& nfex) {
        val = false;
    }
    return val;
}

void Configuration::get_storage_classes(std::vector<unsigned long long int>* capacities,
                                        std::vector<int>* threads,
                                        std::vector<std::map<int, int>>* bandwidths,
                                        std::vector<std::string>* pf_backends,
                                        std::vector<std::map<std::string, std::string>>* pf_backend_options) {
    const libconfig::Setting& root = cfg.getRoot();
    const libconfig::Setting& storage_classes = root["storage_classes"];
    int count = storage_classes.getLength();
    for (int i = 0; i < count; i++) {
        const libconfig::Setting& storage_class = storage_classes[i];
        int config_capacity = storage_class.lookup("capacity");
        unsigned long long capacity = config_capacity;
        capacities->push_back(capacity * 1024 * 1024);
        int no_threads = storage_class.lookup("threads");
        threads->push_back(no_threads);
        const libconfig::Setting& bandwidth = storage_class.lookup("bandwidth");
        int bw_count = bandwidth.getLength();
        std::map<int, int> bw_mappings;
        for (int j = 0; j < bw_count; j++) {
            int bw_threads = bandwidth[j].lookup("threads");
            int bw_bw = bandwidth[j].lookup("bw");
            bw_mappings[bw_threads] = bw_bw;
        }
        bandwidths->push_back(bw_mappings);
        if (i > 0) {
            std::string backend = storage_class.lookup("backend");
            pf_backends->push_back(backend);
            std::map<std::string, std::string> backend_options_map;
            if (storage_class.exists("backend_options")) {
                libconfig::Setting& backend_options = storage_class.lookup("backend_options");
                for (int j = 0; j < backend_options.getLength(); j++) {
                    const libconfig::Setting& child = backend_options[j];
                    backend_options_map[child.getName()] = backend_options.lookup(child.getName()).c_str();
                }
            }
            pf_backend_options->push_back(backend_options_map);
        }
    }
}

void Configuration::get_pfs_bandwidth(std::map<int, int>* bandwidths) {
    const libconfig::Setting& root = cfg.getRoot();
    const libconfig::Setting& pfs_bandwidth = root["pfs_bandwidth"];
    int count = pfs_bandwidth.getLength();
    for (int i = 0; i < count; i++) {
        const libconfig::Setting& bw = pfs_bandwidth[i];
        int processes = bw.lookup("processes");
        int bandwidth = bw.lookup("bw");
        (*bandwidths)[processes] = bandwidth;
    }
}

int Configuration::get_no_distributed_threads() {
    return get_int_entry("distributed_threads");
}

void Configuration::get_bandwidths(int* networkbandwidth_clients, int* networkbandwith_filesystem) {
    *networkbandwidth_clients = get_int_entry("b_c");
    *networkbandwith_filesystem = get_int_entry("b_fs");
}

bool Configuration::get_checkpoint() {
    return get_bool_entry("checkpoint");
}

std::string Configuration::get_checkpoint_path() {
    return get_string_entry("checkpoint_path");
}

bool Configuration::get_profiling() const {
    return profiling_enabled;
}
