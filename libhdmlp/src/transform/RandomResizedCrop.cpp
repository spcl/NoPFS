#include <random>
#include "../../include/transform/RandomResizedCrop.h"

char* RandomResizedCrop::parse_arguments(char* arg_array) {
    size = *((int*) arg_array);
    arg_array += sizeof(int);
    scale_lb = *((float*) arg_array);
    arg_array += sizeof(float);
    scale_ub = *((float*) arg_array);
    arg_array += sizeof(float);
    ratio_lb = *((float*) arg_array);
    arg_array += sizeof(float);
    ratio_ub = *((float*) arg_array);
    arg_array += sizeof(float);
    return arg_array;
}

void RandomResizedCrop::transform(TransformPipeline* pipeline) {
    cv::Mat out_image;
    int img_width = pipeline->img.cols;
    int img_height = pipeline->img.rows;
    int img_area = img_width * img_height;
    int w = -1, h = -1, i = -1, j = -1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_scale(scale_lb, scale_ub);
    std::uniform_real_distribution<> dis_ratio(log(ratio_lb), log(ratio_ub)); // We draw exp([log(lb), log(ub)]) as in torchvision
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        double scale = dis_scale(gen);
        double ratio = exp(dis_ratio(gen));
        double target_area = scale * img_area;
        w = (int) sqrt(target_area * ratio);
        h = (int) sqrt(target_area / ratio);
        if (w > 0 && w <= img_width && h > 0 && h <= img_height) {
            std::uniform_int_distribution<> dis_i(0, img_height - h);
            i = dis_i(gen);
            std::uniform_int_distribution<> dis_j(0, img_width - w);
            j = dis_j(gen);
            break;
        }
    }
    if (i == -1) {
        // Fallback to center crop
        float img_ratio = (float) img_width / (float) img_height;
        if (img_ratio < ratio_lb) {
            w = img_width;
            h = (int) ((float) w / img_ratio);
        } else if (img_ratio > ratio_ub) {
            h = img_height;
            w = (int) ((float) h * ratio_ub);
        } else {
            w = img_width;
            h = img_height;
        }
        i = (img_height - h) / 2;
        j = (img_width - w) / 2;
    }
    cv::Rect roi(j, i, w, h);
    cv::Mat cropped(pipeline->img, roi);
    cv::resize(cropped, cropped, cv::Size(size, size));
    cropped.copyTo(pipeline->img);
}