#ifndef HDMLP_IMGDECODE_H
#define HDMLP_IMGDECODE_H


#include "Transformation.h"

class ImgDecode : public Transformation {
public:
    void transform(TransformPipeline* pipeline);

};


#endif //HDMLP_IMGDECODE_H
