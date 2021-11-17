#ifndef HDMLP_TOTENSOR_H
#define HDMLP_TOTENSOR_H


#include "Transformation.h"

class ToTensor : public Transformation {
public:
    void transform(TransformPipeline* pipeline);

};


#endif //HDMLP_TOTENSOR_H
