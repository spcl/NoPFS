#include "../../include/transform/ImgDecode.h"

void ImgDecode::transform(TransformPipeline* pipeline) {
    pipeline->img = cv::imdecode(cv::Mat(1, (int) pipeline->src_len, CV_8UC1, pipeline->src_buffer), cv::IMREAD_COLOR);
}
