#ifndef HDMLP_NORMALIZE_H
#define HDMLP_NORMALIZE_H


#include "Transformation.h"

class Normalize : public Transformation {
public:
    char* parse_arguments(char* arg_array);

    void transform(TransformPipeline* pipeline);

private:
double mean[3];
double std[3];

};


#endif //HDMLP_NORMALIZE_H
