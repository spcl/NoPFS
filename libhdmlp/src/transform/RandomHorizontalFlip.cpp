#include <random>
#include "../../include/transform/RandomHorizontalFlip.h"

char* RandomHorizontalFlip::parse_arguments(char* arg_array) {
    p = *((float*) arg_array);
    return arg_array + sizeof(float);
}

void RandomHorizontalFlip::transform(TransformPipeline* pipeline) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::bernoulli_distribution d(p);
    if (d(rng)) {
        cv::flip(pipeline->img, pipeline->img, 1);
    }
}
