#include <codecvt>
#include "../../include/transform/Transformation.h"
#include "../../include/transform/ImgDecode.h"
#include "../../include/transform/Resize.h"
#include "../../include/transform/ToTensor.h"
#include "../../include/transform/RandomHorizontalFlip.h"
#include "../../include/transform/RandomVerticalFlip.h"
#include "../../include/transform/RandomResizedCrop.h"
#include "../../include/transform/Normalize.h"
#include "../../include/transform/CenterCrop.h"
#include "../../include/transform/Reshape.h"
#include "../../include/transform/ScaleShift16.h"


TransformPipeline::TransformPipeline(wchar_t** transform_names, char* transform_args, int transform_output_size, int transform_len) {
    output_size = transform_output_size;
    for (int i = 0; i < transform_len; i++) {
        using type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<type, wchar_t> converter;
        std::string transform_name = converter.to_bytes(transform_names[i]);
        Transformation* transform = nullptr;
        if (transform_name == "ImgDecode") {
            transform = new ImgDecode();
        } else if (transform_name == "Resize") {
            transform = new Resize();
        } else if (transform_name == "ToTensor") {
            transform = new ToTensor();
        } else if (transform_name == "RandomHorizontalFlip") {
            transform = new RandomHorizontalFlip();
        } else if (transform_name == "RandomVerticalFlip") {
            transform = new RandomVerticalFlip();
        } else if (transform_name == "RandomResizedCrop") {
            transform = new RandomResizedCrop();
        } else if (transform_name == "Normalize") {
            transform = new Normalize();
        } else if (transform_name == "CenterCrop") {
            transform = new CenterCrop();
        } else if (transform_name == "Reshape") {
            transform = new Reshape();
        } else if (transform_name == "ScaleShift16") {
            transform = new ScaleShift16();
        } else if (transform_name == "HWCtoCHW") {
          hwc_to_chw = true;
        } else {
            throw std::runtime_error("Transformation not implemented");
        }
	if (transform) {
	    transform_args = transform->parse_arguments(transform_args);
	    transformations.push_back(transform);
	    transformation_names.push_back(transform_name);
	}
    }
}

void TransformPipeline::transform(char* src_buffer, unsigned long src_len, char* dst_buffer) {
    this->src_buffer = src_buffer;
    this->src_len = src_len;
    for (auto const &transformation : transformations) {
        transformation->transform(this);
    }
    // Hack, but this avoids a memory copy.
    if (hwc_to_chw) {
      do_hwc_to_chw(dst_buffer);
    } else {
      memcpy(dst_buffer, img.data, output_size);
    }
}

TransformPipeline::~TransformPipeline() {
    for (auto &transformation : transformations) {
        delete transformation;
    }
}

int TransformPipeline::get_output_size() const {
    return output_size;
}

void TransformPipeline::do_hwc_to_chw(char* dst_buffer) {
  const uint8_t* __restrict__ src_buf = img.ptr();
  const int height = img.rows;
  const int width = img.cols;
  const int size = height * width;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      // Three channels.
      const size_t src_base = 3 * (row*width + col);
      const size_t dst_base = row + col*height;
      dst_buffer[dst_base] = src_buf[src_base];
      dst_buffer[dst_base + size] = src_buf[src_base + 1];
      dst_buffer[dst_base + 2*size] = src_buf[src_base + 2];
    }
  }
}
