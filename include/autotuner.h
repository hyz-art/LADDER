#pragma once
#include <vector>
#include <cstddef>

namespace ladder {

struct TuningResult {
    size_t tileM, tileN, tileK;
    double ms;
};

class AutoTuner {
public:
    // run autotuner for given matrix dims and candidate tiles; returns best result
    TuningResult tuneGEMM(size_t M, size_t N, size_t K,
                          const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                          const std::vector<size_t>& tileMs,
                          const std::vector<size_t>& tileNs,
                          const std::vector<size_t>& tileKs,
                          const std::vector<float>* bias = nullptr,
                          bool apply_relu = false,
                          int repeats = 3);
};

} // namespace ladder
