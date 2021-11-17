#include "../../include/transform/ScaleShift16.h"

char* ScaleShift16::parse_arguments(char* arg_array) {
    for (int i = 0; i < 16; i++) {
        shift[i] = *((float*) arg_array + i);
    }
    for (int i = 0; i < 16; i++) {
        scale[i] = *((float*) arg_array + 16 + i);
    }
    return arg_array + 32 * sizeof(float);
}

void ScaleShift16::transform(TransformPipeline* pipeline) {
    for(int row = 0; row < pipeline->img.rows; ++row) {
        for(int col = 0; col < pipeline->img.cols; ++col) {
            pipeline->img.at<cv::Vec<float, 16>>(row, col) -= shift;
            pipeline->img.at<cv::Vec<float, 16>>(row, col) = pipeline->img.at<cv::Vec<float, 16>>(row, col).mul(shift);
        }

    }
}
