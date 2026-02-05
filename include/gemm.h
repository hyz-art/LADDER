#pragma once
#include <vector>
#include <cstddef>
#include "ttype.h"

namespace ladder {

// Simple GEMM prototypes: A (M x K) * B (K x N) = C (M x N)
void gemm_naive(size_t M, size_t N, size_t K,
                const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C);

// Tiled GEMM with simple fusion (bias add + ReLU) and tile size parameters
void gemm_tiled_fused(size_t M, size_t N, size_t K,
                      const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                      size_t tileM, size_t tileN, size_t tileK,
                      const std::vector<float>* bias = nullptr,
                      bool apply_relu = false,
                      tType::Precision precision = tType::Precision::FP32);

// Tile-level variant that uses tTile objects for per-tile operations
void gemm_tiled_fused_tiles(size_t M, size_t N, size_t K,
                           const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                           size_t tileM, size_t tileN, size_t tileK,
                           const std::vector<float>* bias = nullptr,
                           bool apply_relu = false,
                           tType::Precision precision = tType::Precision::FP32);

} // namespace ladder
