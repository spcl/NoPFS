#include "../../include/transform/ToTensor.h"

void ToTensor::transform(TransformPipeline* pipeline) {
    cv::Mat float_img;
    pipeline->img.convertTo(float_img, CV_32F, 1./255.);
    float_img.copyTo(pipeline->img);
}
