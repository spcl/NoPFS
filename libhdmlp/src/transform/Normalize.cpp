#include "../../include/transform/Normalize.h"

char* Normalize::parse_arguments(char* arg_array) {
    double* args = ((double*) arg_array);
    mean[0] = *args;
    mean[1] = *(args + 1);
    mean[2] = *(args + 2);
    std[0] = *(args + 3);
    std[1] = *(args + 4);
    std[2] = *(args + 5);
    return arg_array + 6 * sizeof(double);
}

void Normalize::transform(TransformPipeline* pipeline) {
    cv::subtract(pipeline->img, cv::Scalar(mean[0], mean[1], mean[2]), pipeline->img);
    cv::multiply(pipeline->img, cv::Scalar(1. / std[0], 1. / std[1], 1. / std[2]), pipeline->img);
}
