#pragma once

#include <cstddef>
#include <vector>
#include "ttype.h"
#include "gemm.h"

namespace ladder {

// Returns true if CUDA path ran successfully; false to fall back to CPU.
bool gemm_tiled_fused_tiles_cuda(size_t M, size_t N, size_t K,
                                const std::vector<float>& A,
                                const std::vector<float>& B,
                                std::vector<float>& C,
                                const std::vector<float>* bias,
                                bool apply_relu,
                                tType::Precision precision);

bool gemm_tiled_fused_tiles_quantized_cuda(size_t M, size_t N, size_t K,
                                           const std::vector<float>& A,
                                           const std::vector<float>& B,
                                           std::vector<float>& C,
                                           const QuantParams& q,
                                           const std::vector<float>* bias,
                                           bool apply_relu);

} // namespace ladder
