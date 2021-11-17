#include "../../include/transform/Resize.h"

char* Resize::parse_arguments(char* arg_array) {
    width = *((int*) arg_array);
    height = *((int*) (arg_array + sizeof(int)));
    return arg_array + 2 * sizeof(int);
}

void Resize::transform(TransformPipeline* pipeline) {
    cv::resize(pipeline->img, pipeline->img, cv::Size(width, height));
}