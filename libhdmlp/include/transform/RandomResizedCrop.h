#ifndef HDMLP_RANDOMRESIZEDCROP_H
#define HDMLP_RANDOMRESIZEDCROP_H

#define MAX_ATTEMPTS 10

#include "Transformation.h"

class RandomResizedCrop : public Transformation {
public:
    char* parse_arguments(char* arg_array);

    void transform(TransformPipeline* pipeline);

private:
    int size;
    float scale_lb, scale_ub, ratio_lb, ratio_ub;
};


#endif //HDMLP_RANDOMRESIZEDCROP_H
