#ifndef HDMLP_TRANSFORMATION_H
#define HDMLP_TRANSFORMATION_H
#include "TransformPipeline.h"


class Transformation {
public:
    virtual ~Transformation() = 0;
    // Parse the required arguments out of the arg_array (pointing to the offset of the transformation, return new pointer containing offset of next transformation)
    virtual char* parse_arguments(char* arg_array);

    virtual void transform(TransformPipeline* pipeline) = 0;
};


#endif //HDMLP_TRANSFORMATION_H
