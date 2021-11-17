#ifndef HDMLP_CENTERCROP_H
#define HDMLP_CENTERCROP_H

#include "Transformation.h"

class CenterCrop : public Transformation {
public:
    char* parse_arguments(char* arg_array);

    void transform(TransformPipeline* pipeline);

private:
    int height;
    int width;
};


#endif //HDMLP_CENTERCROP_H
