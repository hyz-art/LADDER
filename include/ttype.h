#pragma once
#include <string>

namespace ladder {

class tType {
public:
    enum class Precision {
        FP32,
        FP16,
        FP8,
        INT8,
        INT4
    };

    tType(Precision p = Precision::FP32);
    Precision precision() const;
    std::string toString() const;
    // simple conversion utility for demo purposes
    static float convertToFloat(int v, Precision p);
private:
    Precision p_;
};

} // namespace ladder

// 保持向后兼容：允许使用 ladder::Precision
namespace ladder {
using Precision = tType::Precision;
}
