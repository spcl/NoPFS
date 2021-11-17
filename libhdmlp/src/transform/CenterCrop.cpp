#include "../../include/transform/CenterCrop.h"

char* CenterCrop::parse_arguments(char* arg_array) {
    width = *((int*) arg_array);
    height = *((int*) (arg_array + sizeof(int)));
    return arg_array + 2 * sizeof(int);
}

void CenterCrop::transform(TransformPipeline* pipeline) {
    int img_width = pipeline->img.cols;
    int img_height = pipeline->img.rows;
    int i = (img_height - height) / 2;
    int j = (img_width - width) / 2;
    cv::Rect roi(j, i, width, height);
    cv::Mat cropped(pipeline->img, roi);
    cropped.copyTo(pipeline->img);
}
