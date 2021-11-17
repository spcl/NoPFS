#ifndef HDMLP_RANDOMVERTICALFLIP_H
#define HDMLP_RANDOMVERTICALFLIP_H


#include "Transformation.h"

class RandomVerticalFlip : public Transformation {
public:
    char* parse_arguments(char* arg_array);

    void transform(TransformPipeline* pipeline);

private:
    float p;
};


#endif //HDMLP_RANDOMVERTICALFLIP_H
