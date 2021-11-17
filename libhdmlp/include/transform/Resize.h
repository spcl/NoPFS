#ifndef HDMLP_RESIZE_H
#define HDMLP_RESIZE_H


#include "Transformation.h"

class Resize : public Transformation {
public:
    char* parse_arguments(char* arg_array);

    void transform(TransformPipeline* pipeline);

private:
    int height;
    int width;
};


#endif //HDMLP_RESIZE_H
