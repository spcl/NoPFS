#ifndef HDMLP_RESHAPE_H
#define HDMLP_RESHAPE_H


#include "Transformation.h"

class Reshape : public Transformation {
public:
    char* parse_arguments(char* arg_array);

    void transform(TransformPipeline* pipeline);

private:
    int w, h, c;
};


#endif //HDMLP_RESHAPE_H
