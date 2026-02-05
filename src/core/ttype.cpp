#include "ttype.h"
#include <stdexcept>

namespace ladder {

tType::tType(Precision p) : p_(p) {}

tType::Precision tType::precision() const { return p_; }

std::string tType::toString() const {
    switch (p_) {
        case Precision::FP32: return "FP32";
        case Precision::FP16: return "FP16";
        case Precision::FP8:  return "FP8";
        case Precision::INT8: return "INT8";
        case Precision::INT4: return "INT4";
    }
    throw std::runtime_error("Unknown precision");
}

float tType::convertToFloat(int v, tType::Precision p) {
    // prototype conversion; real implementation would use scaling/quantization
    switch (p) {
        case tType::Precision::FP32: return static_cast<float>(v);
        case tType::Precision::FP16: return static_cast<float>(v) * 0.5f;
        case tType::Precision::FP8:  return static_cast<float>(v) * 0.25f;
        case tType::Precision::INT8: return static_cast<float>(v) / 2.0f;
        case tType::Precision::INT4: return static_cast<float>(v) / 4.0f;
    }
    return static_cast<float>(v);
}

} // namespace ladder
