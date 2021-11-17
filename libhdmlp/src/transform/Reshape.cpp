#include "../../include/transform/Reshape.h"


char* Reshape::parse_arguments(char* arg_array) {
    w = *((int*) arg_array);
    h = *((int*) (arg_array + sizeof(int)));
    c = *((int*) (arg_array + 2 * sizeof(int)));

    return arg_array + 3 * sizeof(int);
}


void Reshape::transform(TransformPipeline* pipeline) {
    pipeline->img = cv::Mat(cv::Size(w, h), CV_32FC(c), pipeline->src_buffer);
}
