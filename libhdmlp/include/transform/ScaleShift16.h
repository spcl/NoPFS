#ifndef HDMLP_SCALESHIFT16_H
#define HDMLP_SCALESHIFT16_H


#include "Transformation.h"

class ScaleShift16 : public Transformation {
public:
    char* parse_arguments(char* arg_array);

    void transform(TransformPipeline* pipeline);

private:
    cv::Vec<float, 16> scale;
    cv::Vec<float, 16> shift;
};


#endif //HDMLP_SCALESHIFT16_H
