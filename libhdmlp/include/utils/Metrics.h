#ifndef HDMLP_METRICS_H
#define HDMLP_METRICS_H

#include <vector>

struct Metrics {
    std::vector<double> stall_time;
    std::vector<std::vector<double>> augmentation_time; // Values per thread
    std::vector<std::vector<std::vector<double>>> read_times; // Values per storage class * thread
    std::vector<std::vector<std::vector<int>>> read_locations; // Values per storage class * thread
};

#endif //HDMLP_METRICS_H
